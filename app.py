from __future__ import annotations
import hashlib
import io
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError, ImageStat
from pydantic import BaseModel, Field
from auth import AuthStore, COOKIE, OWNER_USERNAME
from domain import Report, ReviewPatch, apply_patch, rebuild, review_errors, now
from storage import Store
from vision import Vision, crop_box
from retrieval import ReferenceIndex
from pipeline import Workflow

ROOT = Path(__file__).resolve().parent
Image.MAX_IMAGE_PIXELS = 24_000_000


class NewDemo(BaseModel):
    sensitivity: bool = False


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str
    new_password: str | None = None


class VLMRequest(BaseModel):
    consent: bool = False


def create_app(data_dir=None, vision=None, references=None, auth_enabled=True):
    data_dir = Path(data_dir or os.getenv('ORTHOGRAPH_DATA_DIR',str(ROOT/'data')))
    auth = AuthStore(data_dir) if auth_enabled else None
    store = Store(data_dir)
    vision = vision or Vision()
    references = references or ReferenceIndex(data_dir)
    workflow = Workflow(store,vision,references)
    executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='orthograph-gpu')
    slots = threading.BoundedSemaphore(4)
    edit_lock = threading.RLock()
    vlm_slots = threading.BoundedSemaphore(1)

    @asynccontextmanager
    async def lifespan(app):
        yield
        executor.shutdown(wait=True)
        references.close()
        workflow.connection.close()
        store.close()
        if auth: auth.close()

    app = FastAPI(title='OrthoGraph',version='0.1.0',lifespan=lifespan)
    app.state.store,app.state.workflow = store,workflow

    @app.middleware('http')
    async def access_control(request, call_next):
        path = request.url.path
        public = path in ('/login', '/api/login', '/register', '/api/register')
        user = auth.user_for_token(request.cookies.get(COOKIE)) if auth else {'id': 1, 'username': OWNER_USERNAME}
        request.state.user = user
        if auth and not public and not user:
            if path == '/' and request.method == 'GET':
                return JSONResponse(status_code=303, content={}, headers={'Location': '/login', 'Cache-Control': 'no-store'})
            return JSONResponse({'detail': 'Sign in to continue.'}, status_code=401, headers={'Cache-Control': 'no-store'})
        if request.method in ('POST', 'PATCH', 'DELETE'):
            if request.headers.get('sec-fetch-site') == 'cross-site':
                return JSONResponse({'detail': 'Cross-site requests are not allowed.'}, status_code=403)
            origin = request.headers.get('origin')
            expected_scheme = request.headers.get('x-forwarded-proto', request.url.scheme).split(',')[0].strip()
            expected_origin = f'{expected_scheme}://{request.headers.get("host", request.url.netloc)}'
            if origin and origin != expected_origin:
                return JSONResponse({'detail': 'Invalid request origin.'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['Cache-Control'] = 'no-store' if path.startswith('/api') or path in ('/login', '/register') else 'no-cache'
        return response

    @app.get('/login')
    def login_page():
        return FileResponse(ROOT / 'static/login.html')

    @app.get('/register')
    def register_page():
        return FileResponse(ROOT / 'static/register.html')

    def session_response(token, user, request):
        response = JSONResponse({'username': user['username']})
        secure = request.url.scheme == 'https' or request.headers.get('x-forwarded-proto') == 'https'
        response.set_cookie(COOKIE, token, max_age=7 * 24 * 3600, httponly=True, secure=secure, samesite='strict', path='/')
        return response

    @app.post('/api/register', status_code=201)
    def register(body: RegisterRequest, request: Request):
        if not auth:
            raise HTTPException(404, 'Authentication is disabled in this test instance.')
        result, error = auth.register(body.username, body.password, request.client.host if request.client else 'unknown')
        if error:
            raise HTTPException(429 if error.startswith('Too many') else 400, error)
        response = session_response(*result, request)
        response.status_code = 201
        return response

    @app.post('/api/login')
    def login(body: LoginRequest, request: Request):
        if not auth:
            raise HTTPException(404, 'Authentication is disabled in this test instance.')
        result, error = auth.authenticate(body.username, body.password, body.new_password, request.client.host if request.client else 'unknown')
        if error == 'SET_PASSWORD_REQUIRED':
            return JSONResponse({'detail': 'Choose a new password to finish account setup.', 'code': 'set_password_required'}, status_code=428)
        if error:
            raise HTTPException(429 if error.startswith('Too many') else 401, error)
        return session_response(*result, request)

    @app.get('/api/session')
    def session(request: Request):
        user = request.state.user
        return {'username': user['username'], 'setup_required': auth.needs_setup(user['id']) if auth else False}

    @app.post('/api/logout')
    def logout(request: Request):
        if auth:
            auth.logout(request.cookies.get(COOKIE))
        response = JSONResponse({'signed_out': True})
        response.delete_cookie(COOKIE, path='/')
        return response

    def get_case(case_id, request):
        try:
            return store.get(case_id, request.state.user['id'])
        except KeyError:
            raise HTTPException(404,'Case not found.') from None

    def get_report(case_id, request):
        case = get_case(case_id, request)
        if not case['report']:
            raise HTTPException(409,'This case is still processing.')
        return Report(**case['report'])

    def enqueue(raw, name, request, orientation='unknown', impaction=True, sensitivity=False):
        if orientation not in ('standard','flipped','unknown'):
            raise HTTPException(422,'Invalid orientation.')
        if not slots.acquire(blocking=False):
            raise HTTPException(429,'The analysis queue is full. Please try again shortly.')
        try:
            if len(raw)>20*1024*1024:
                raise HTTPException(413,'Image must be smaller than 20 MB.')
            try:
                with Image.open(io.BytesIO(raw)) as source:
                    if source.format not in ('PNG','JPEG'):
                        raise ValueError('Use a PNG or JPEG panoramic image.')
                    image = ImageOps.exif_transpose(source).convert('RGB')
                    image.load()
            except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError):
                raise HTTPException(422,'Could not read this image. Use a PNG or JPEG under 24 megapixels.') from None
            if image.width<400 or image.height<200:
                raise HTTPException(422,'Image is too small. Use the original panoramic export (at least 400 × 200).')
            if max(ImageStat.Stat(image).stddev)<3:
                raise HTTPException(422,'The image contains too little contrast to analyze.')
            case_id = str(uuid4())
            directory = store.directory(case_id)
            directory.mkdir(parents=True)
            image.save(directory/'image.png')
            image_hash = hashlib.sha256((directory/'image.png').read_bytes()).hexdigest()
            if request.state.user['id'] != 1 and not store.reserve_analysis(request.state.user['id']):
                shutil.rmtree(directory)
                raise HTTPException(429, 'This demo allows three analyses per account each day.')
            store.create(case_id,Path(name).name[:100],{'orientation':orientation,'impaction':impaction,
                                                       'sensitivity':sensitivity,
                                                       'image_hash':image_hash}, owner_id=request.state.user['id'])
            def run():
                try:
                    workflow.run(case_id)
                finally:
                    slots.release()
            executor.submit(run)
            return store.get(case_id)
        except Exception:
            slots.release()
            raise

    @app.get('/api/health')
    def health():
        import torch
        return {'status':'ok','version':'0.1.0','device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'gemini_configured':bool(os.getenv('GEMINI_API_KEY')),'retrieval':references.status,
                'models_loaded':list(vision.models),'corpus_size':len(references.records)}

    @app.get('/api/cases')
    def cases(request: Request):
        return store.list(request.state.user['id'])

    @app.post('/api/cases',status_code=202)
    async def upload(request: Request, image:UploadFile=File(...),orientation:str=Form('unknown'),impaction:bool=Form(True),
                     sensitivity:bool=Form(False)):
        raw = await image.read(20*1024*1024+1)
        return enqueue(raw,image.filename or 'Panoramic image',request,orientation,impaction,sensitivity)

    @app.post('/api/demo',status_code=202)
    def demo(request: Request, body:NewDemo=NewDemo()):
        path = ROOT/'research/sample.jpg'
        if not path.exists():
            raise HTTPException(404,'Demo image unavailable. Upload an authorized OPG.')
        return enqueue(path.read_bytes(),'Research example · panoramic OPG',request,'standard',sensitivity=body.sensitivity)

    @app.get('/api/cases/{case_id}')
    def case(case_id:str, request: Request):
        return get_case(case_id, request)

    @app.get('/api/cases/{case_id}/image')
    def case_image(case_id:str, request: Request):
        get_case(case_id, request)
        return FileResponse(store.directory(case_id)/'image.png',media_type='image/png')

    @app.get('/api/cases/{case_id}/crop/{finding_id}')
    def crop(case_id:str,finding_id:str, request: Request):
        report = get_report(case_id, request)
        finding = next((f for f in report.findings if f.id==finding_id),None)
        tooth = next((t for t in report.teeth if t.id==finding_id),None)
        item = finding or tooth
        if item is None:
            raise HTTPException(404,'Region not found.')
        image,_ = crop_box(store.directory(case_id)/'image.png',item.bbox)
        buffer=io.BytesIO();image.save(buffer,format='PNG')
        return Response(buffer.getvalue(),media_type='image/png')

    @app.patch('/api/cases/{case_id}/review')
    def review(case_id:str,patch:ReviewPatch, request: Request):
        with edit_lock:
            report = get_report(case_id, request)
            if get_case(case_id, request)['status']!='awaiting_review':
                raise HTTPException(409,'This case is not available for editing.')
            try:
                report = apply_patch(report,patch)
            except ValueError as exc:
                raise HTTPException(409,str(exc)) from None
            references.attach(report)
            rebuild(report)
            store.update(case_id,report=report.model_dump())
            return store.get(case_id)

    @app.post('/api/cases/{case_id}/finalize')
    def finalize(case_id:str, request: Request):
        with edit_lock:
            report = get_report(case_id, request)
            if get_case(case_id, request)['status']!='awaiting_review':
                raise HTTPException(409,'This case is not waiting for review.')
            errors = review_errors(report)
            if errors:
                raise HTTPException(409,' '.join(errors))
            try:
                return workflow.finish(case_id)
            except Exception:
                raise HTTPException(500,'Could not finalize this review. Your saved edits are preserved.') from None

    @app.post('/api/cases/{case_id}/inspect/{finding_id}')
    def inspect(case_id:str,finding_id:str,body:VLMRequest, request: Request):
        if not body.consent:
            raise HTTPException(422,'Explicit consent to send this crop to Gemini is required.')
        if not vlm_slots.acquire(blocking=False):
            raise HTTPException(429,'Another crop inspection is running. Try again shortly.')
        try:
            with edit_lock:
                report = get_report(case_id, request)
                if report.status=='reviewed':
                    raise HTTPException(409,'This review is finalized.')
                finding = next((f for f in report.findings if f.id==finding_id),None)
                if finding is None:
                    raise HTTPException(404,'Finding not found.')
                if finding.vlm:
                    return store.get(case_id)
                if sum(1 for e in report.trace if e['node']=='vlm_request')>=3:
                    raise HTTPException(429,'This case has reached the three-request crop inspection limit.')
                report.trace.append({'node':'vlm_request','at':now(),'detail':f'User requested Gemini inspection for {finding_id}; crop sent to Google'})
                report.revision+=1
                store.update(case_id,report=report.model_dump())
            from vlm import inspect_crop
            try:
                observation = inspect_crop(store.directory(case_id)/'image.png',finding)
            except RuntimeError as exc:
                raise HTTPException(502,str(exc)) from None
            except Exception:
                raise HTTPException(502,'Crop inspection was unavailable. No finding was changed.') from None
            with edit_lock:
                current = get_report(case_id, request)
                if current.status=='reviewed':
                    raise HTTPException(409,'Review was finalized while the request ran. The observation was not applied.')
                target = next(f for f in current.findings if f.id==finding_id)
                target.vlm = observation
                current.revision+=1
                current.trace.append({'node':'vlm_observation','at':now(),'detail':'Unverified crop observation saved; finding decisions were not changed'})
                store.update(case_id,report=current.model_dump())
                return store.get(case_id)
        finally:
            vlm_slots.release()

    @app.get('/api/cases/{case_id}/export')
    def export(case_id:str, request: Request):
        report=get_report(case_id, request)
        return Response(report.model_dump_json(indent=2),media_type='application/json',
                        headers={'Content-Disposition':f'attachment; filename="orthograph-{case_id[:8]}.json"'})

    @app.delete('/api/cases/{case_id}')
    def delete_case(case_id:str, request: Request):
        with edit_lock:
            case = get_case(case_id, request)
            if case['status'] in ('processing','queued','finalizing'):
                raise HTTPException(409,'Wait for processing to finish before deleting this case.')
            directory = store.directory(case_id)
            workflow.checkpointer.delete_thread(case_id)
            with store.lock:
                store.db.execute('DELETE FROM cases WHERE id=? AND owner_id=?',(case_id,request.state.user['id']))
                store.db.commit()
            if directory.exists():
                shutil.rmtree(directory)
            return {'deleted':case_id}

    @app.get('/api/references')
    def reference_library():
        return references.records

    @app.get('/api/evaluation')
    def evaluation():
        path = ROOT/'research/dentex_validation.json'
        if not path.exists():
            raise HTTPException(404,'DENTEX evaluation has not been run in this workspace.')
        return json.loads(path.read_text())

    @app.get('/api/references/search')
    def reference_search(q:str='panoramic assessment'):
        if len(q)>500:
            raise HTTPException(422,'Search query is too long.')
        return references.search(q,limit=3)

    @app.get('/')
    def home():
        return FileResponse(ROOT/'static/index.html')

    app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')
    return app
