import io
import time
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from app import create_app


class FakeVision:
    models={}
    def predict(self,name,path,include_all=False):
        if name=='enumeration':
            records=[{'id':'t01','tooth_type':1,'bbox':[100,50,160,130],'score':.9}]
        else:
            records=[{'id':'yolo26-f01','label':'filling','title':'Dental filling','category':'restoration',
                      'raw_label':'Dental Filling','model':'yolo26','bbox':[105,55,130,80],'score':.8}]
        return records,{'id':name,'model':name,'predict_ms':1}


class FakeReferences:
    records=[]
    status='test'
    def attach(self,r):return r
    def close(self):pass
    def search(self,*args,**kwargs):return []


def image_bytes():
    im=Image.new('RGB',(800,400),'black')
    for x in range(400):
        for y in range(200):im.putpixel((x,y),(128,128,128))
    buf=io.BytesIO();im.save(buf,format='PNG');return buf.getvalue()


def wait_for_review(client, case_id, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        case = client.get('/api/cases/'+case_id).json()
        if case['status'] == 'awaiting_review':
            return case
        assert case['status'] != 'error', case
        time.sleep(.02)
    raise AssertionError(f'Case did not reach review: {case}')


def test_upload_review_resume_export_and_persistence(tmp_path):
    app=create_app(tmp_path,FakeVision(),FakeReferences(),auth_enabled=False)
    with TestClient(app) as client:
        response=client.post('/api/cases',files={'image':('opg.png',image_bytes(),'image/png')},data={'orientation':'standard'})
        assert response.status_code==202
        cid=response.json()['id']
        c=wait_for_review(client,cid)
        assert c['status']=='awaiting_review',c
        assert client.post(f'/api/cases/{cid}/finalize').status_code==409
        assert client.get(f'/api/cases/{cid}/crop/yolo26-f01').status_code==200
        patch={'expected_revision':0,'orientation_confirmed':True,'numbering_confirmed':True,
               'finding_reviews':{'yolo26-f01':'accepted'}}
        response=client.patch(f'/api/cases/{cid}/review',json=patch)
        assert response.status_code==200,response.text
        assert client.patch(f'/api/cases/{cid}/review',json=patch).status_code==409
        checkpoint=app.state.workflow.graph.get_state({'configurable':{'thread_id':cid}})
        assert checkpoint.next==('review',)
    # Actual SQLite checkpoint resume across an application restart.
    with TestClient(create_app(tmp_path,FakeVision(),FakeReferences(),auth_enabled=False)) as client:
        response=client.post(f'/api/cases/{cid}/finalize')
        assert response.status_code==200,response.text
        assert response.json()['report']['status']=='reviewed'
        exported=client.get(f'/api/cases/{cid}/export')
        assert exported.status_code==200 and exported.json()['findings'][0]['review']=='accepted'
        assert client.patch(f'/api/cases/{cid}/review',json=patch).status_code==409
        assert client.post(f'/api/cases/{cid}/finalize').status_code==409


def test_upload_and_external_inspection_validation(tmp_path):
    with TestClient(create_app(tmp_path,FakeVision(),FakeReferences(),auth_enabled=False)) as client:
        assert client.post('/api/cases',files={'image':('fake.png',b'not an image','image/png')}).status_code==422
        assert client.post('/api/cases',files={'image':('valid.png',image_bytes(),'image/png')},data={'orientation':'bad'}).status_code==422
        assert client.get('/api/cases/not-a-real-id').status_code==404
        assert client.post('/api/cases/x/inspect/y',json={'consent':False}).status_code==422
        assert client.post('/api/demo',headers={'Sec-Fetch-Site':'cross-site'}).status_code==403


def test_16bit_png_is_scaled_before_detection_and_reported(tmp_path):
    # Pillow's plain I;16 -> RGB conversion clips values above 255 to white.
    gradient = np.linspace(900, 3500, 800, dtype=np.uint16)[None, :].repeat(400, axis=0)
    buffer = io.BytesIO()
    Image.fromarray(gradient).save(buffer, format='PNG')
    with TestClient(create_app(tmp_path, FakeVision(), FakeReferences(), auth_enabled=False)) as client:
        response = client.post('/api/cases', files={'image':('opg-16bit.png', buffer.getvalue(), 'image/png')})
        assert response.status_code == 202, response.text
        cid = response.json()['id']
        case = wait_for_review(client,cid)
        assert case['status'] == 'awaiting_review'
        assert case['report']['image']['source_mode'] == 'I;16'
        assert case['report']['image']['pixel_normalization'] == 'percentile_0.5_99.5_to_8bit'
        with Image.open(io.BytesIO(client.get(f'/api/cases/{cid}/image').content)) as normalized:
            low, high = normalized.getextrema()[0]
            assert low <= 5 and high >= 250


def test_nearly_blank_image_is_rejected_before_inference(tmp_path):
    image = Image.new('RGB', (800, 400), 'white')
    for x in range(280, 520):
        image.putpixel((x, 200), (0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    with TestClient(create_app(tmp_path, FakeVision(), FakeReferences(), auth_enabled=False)) as client:
        response = client.post('/api/cases', files={'image':('mask.png', buffer.getvalue(), 'image/png')})
        assert response.status_code == 422
        assert 'radiographic detail' in response.json()['detail']
        assert client.get('/api/cases').json() == []


def test_delete_removes_image_report_and_checkpoint(tmp_path):
    with TestClient(create_app(tmp_path,FakeVision(),FakeReferences(),auth_enabled=False)) as client:
        cid=client.post('/api/cases',files={'image':('opg.png',image_bytes(),'image/png')}).json()['id']
        wait_for_review(client,cid)
        assert client.delete('/api/cases/'+cid).status_code==200
        assert client.get('/api/cases/'+cid).status_code==404
        assert not (tmp_path/'cases'/cid).exists()
        checkpoint=client.app.state.workflow.graph.get_state({'configurable':{'thread_id':cid}})
        assert not checkpoint.values


def test_sign_in_setup_session_logout_and_origin(tmp_path):
    with TestClient(create_app(tmp_path, FakeVision(), FakeReferences())) as client:
        assert client.get('/api/cases').status_code == 401
        assert client.get('/', follow_redirects=False).status_code == 303
        assert client.get('/login').status_code == 200
        setup = (tmp_path / 'initial-login.txt').read_text()
        temporary = setup.split('Temporary password: ')[1].splitlines()[0]
        assert client.post('/api/login', json={'username':'mohamed-yossri','password':'wrong'}).status_code == 401
        assert client.post('/api/login', json={'username':'mohamed-yossri','password':temporary}).status_code == 428
        login = client.post('/api/login', json={'username':'mohamed-yossri','password':temporary,'new_password':'a sufficiently long password'})
        assert login.status_code == 200
        assert 'httponly' in login.headers['set-cookie'].lower()
        assert not (tmp_path / 'initial-login.txt').exists()
        assert client.get('/api/cases').status_code == 200
        assert client.post('/api/demo', headers={'Origin':'https://evil.example'}).status_code == 403
        assert client.post('/api/logout').status_code == 200
        assert client.get('/api/cases').status_code == 401
        assert client.post('/api/login', json={'username':'mohamed-yossri','password':temporary}).status_code == 401
        assert client.post('/api/login', json={'username':'mohamed-yossri','password':'a sufficiently long password'}).status_code == 200


def test_sensitivity_mode_runs_second_detector_without_third_molar(tmp_path):
    with TestClient(create_app(tmp_path,FakeVision(),FakeReferences(),auth_enabled=False)) as client:
        response=client.post('/api/cases',files={'image':('opg.png',image_bytes(),'image/png')},
                             data={'orientation':'standard','sensitivity':'true'})
        cid=response.json()['id']
        case=wait_for_review(client,cid)
        assert case['status']=='awaiting_review'
        assert [run['model'] for run in case['report']['model_runs']]==['enumeration','yolo26','liodon']
        assert case['options']['sensitivity'] is True


def test_signup_signin_and_case_isolation(tmp_path):
    with TestClient(create_app(tmp_path, FakeVision(), FakeReferences())) as client:
        assert client.get('/register').status_code == 200
        assert client.post('/api/register', json={'username':'alice','password':'short'}).status_code == 400
        created = client.post('/api/register', json={'username':'alice','password':'a strong unique password'})
        assert created.status_code == 201
        assert client.get('/api/session').json()['username'] == 'alice'
        assert client.post('/api/register', json={'username':'ALICE','password':'another strong password'}).status_code == 400
        case = client.post('/api/cases', files={'image':('opg.png',image_bytes(),'image/png')})
        assert case.status_code == 202
        case_id = case.json()['id']
        assert len(client.get('/api/cases').json()) == 1
        assert client.post('/api/logout').status_code == 200
        assert client.get('/api/cases').status_code == 401
        assert client.post('/api/register', json={'username':'bob','password':'a different long password'}).status_code == 201
        assert client.get('/api/cases').json() == []
        for path in (f'/api/cases/{case_id}',f'/api/cases/{case_id}/image',f'/api/cases/{case_id}/export'):
            assert client.get(path).status_code == 404
        assert client.delete('/api/cases/'+case_id).status_code == 404
        assert client.post('/api/logout').status_code == 200
        assert client.post('/api/login', json={'username':'alice','password':'a strong unique password'}).status_code == 200
        assert client.get('/api/cases/'+case_id).status_code == 200


def test_existing_owner_and_cases_migrate(tmp_path):
    import sqlite3
    from argon2 import PasswordHasher
    from auth import AuthStore
    db = sqlite3.connect(tmp_path/'auth.sqlite3')
    db.execute('CREATE TABLE owner (username TEXT PRIMARY KEY,password_hash TEXT NOT NULL,must_change INTEGER NOT NULL)')
    db.execute('CREATE TABLE sessions (token_hash TEXT PRIMARY KEY,expires INTEGER NOT NULL)')
    db.execute('INSERT INTO owner VALUES (?,?,0)',('mohamed-yossri',PasswordHasher().hash('previous long password')))
    db.commit(); db.close()
    from storage import Store
    old = Store(tmp_path)
    old.create('00000000-0000-4000-8000-000000000001','Old case',{})
    old.close()
    with TestClient(create_app(tmp_path, FakeVision(), FakeReferences())) as client:
        assert client.post('/api/login', json={'username':'mohamed-yossri','password':'previous long password'}).status_code == 200
        cases = client.get('/api/cases').json()
        assert len(cases)==1 and cases[0]['name']=='Old case'
        assert client.post('/api/logout').status_code == 200
        assert client.post('/api/register', json={'username':'newuser','password':'another long password'}).status_code == 201
        assert client.get('/api/cases').json() == []


def test_account_can_analyze_more_than_three_cases_in_a_day(tmp_path):
    with TestClient(create_app(tmp_path, FakeVision(), FakeReferences())) as client:
        assert client.post('/api/register', json={'username':'repeatuser','password':'a sufficiently long password'}).status_code == 201
        for _ in range(4):
            response = client.post('/api/cases', files={'image':('opg.png',image_bytes(),'image/png')})
            assert response.status_code == 202
            cid = response.json()['id']
            wait_for_review(client,cid)
            assert client.delete('/api/cases/'+cid).status_code == 200
        assert client.get('/api/cases').json() == []
