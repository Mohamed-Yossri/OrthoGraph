'use strict';
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const base=new URL('.',location.href);
const apiPath=path=>new URL(path.replace(/^\//,''),base).toString();
const state={case:null,selected:null,filter:'all',tab:'report',health:null,zoom:1,panX:0,panY:0,poll:null,busy:false,view:'welcome'};
const statusNames={queued:'Queued',processing:'Processing',awaiting_review:'Awaiting review',reviewed:'Review complete',error:'Analysis interrupted'};
const FDI=[...Array(4)].flatMap((_,q)=>Array.from({length:8},(_,i)=>(q+1)*10+i+1));
let toastTimer;
function toast(message,error=false){$('toast').textContent=message;$('toast').className='toast'+(error?' error':'');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.add('hidden'),error?8500:4200);}
async function api(path,options={}){
 const response=await fetch(apiPath(path),options);
 if(response.status===401){location.replace('/login');throw new Error('Session expired.');}
 if(!response.ok){let data;try{data=await response.json();}catch{}const detail=data?.detail;throw new Error(typeof detail==='string'?detail:Array.isArray(detail)?detail.map(x=>x.msg).join('; '):`Request failed (${response.status}).`);}
 return response.json();
}
function jsonOptions(method,data){return {method,headers:{'Content-Type':'application/json'},body:JSON.stringify(data)};}
function setView(view){state.view=view;for(const id of ['welcome','processing','workspace','library','audit'])$(id).classList.toggle('hidden',id!==view);$('workspaceNav').classList.toggle('active',view!=='library'&&view!=='audit');$('referencesNav').classList.toggle('active',view==='library');$('auditNav').classList.toggle('active',view==='audit');}
async function refreshCases(){
 try{const cases=await api('/api/cases');$('caseList').innerHTML=cases.length?cases.map(c=>`<button class="case-item ${state.case?.id===c.id?'current':''}" data-case="${esc(c.id)}"><strong>${esc(c.name)}</strong><small>${esc(statusNames[c.status]||c.status)} · ${new Date(c.created_at).toLocaleDateString(undefined,{month:'short',day:'numeric'})}</small></button>`).join(''):'<p class="muted">Your cases will appear here.</p>';}catch(e){toast(e.message,true);}
}
function resetZoom(){state.zoom=1;state.panX=0;state.panY=0;updateViewBox();}
function updateViewBox(){if(!state.case?.report)return;const {width:w,height:h}=state.case.report.image;const vw=w/state.zoom,vh=h/state.zoom;$('imageSvg').setAttribute('viewBox',`${(w-vw)/2+state.panX} ${(h-vh)/2+state.panY} ${vw} ${vh}`);$('zoomLabel').textContent=`${Math.round(state.zoom*100)}%`;}
async function openCase(id){
 clearTimeout(state.poll);
 try{state.case=await api(`/api/cases/${id}`);state.selected=null;state.filter='all';resetZoom();renderCase();refreshCases();}catch(e){toast(e.message,true);}
}
function renderCase(){
 const c=state.case;if(!c)return;$('breadcrumbCase').textContent=c.name;
 if(c.status==='error'){setView('welcome');toast(c.error||'Analysis failed. Please try another image.',true);return;}
 if(!c.report||['queued','processing','finalizing'].includes(c.status)){
  setView('processing');$('processStage').textContent=c.stage;
  clearTimeout(state.poll);state.poll=setTimeout(async()=>{try{state.case=await api(`/api/cases/${c.id}`);renderCase();if(state.case.status==='awaiting_review')refreshCases();}catch(e){toast(e.message,true);}},900);return;
 }
 setView('workspace');const r=c.report;const done=r.status==='reviewed';
 $('caseTitle').textContent=c.name;$('caseSubtitle').textContent=`Case ${c.id.slice(0,8).toUpperCase()} · ${new Date(c.created_at).toLocaleString(undefined,{dateStyle:'medium',timeStyle:'short'})} · Adult OPG`;
 $('caseStatus').textContent=statusNames[c.status];$('caseStatus').className='status-badge'+(done?' complete':'');$('exportButton').href=apiPath(`/api/cases/${c.id}/export`);
 $('reviewBanner').querySelector('strong').textContent=done?'Your research review is complete.':'Model findings are ready for your review.';
 $('bannerDetail').textContent=done?' The record preserves original predictions, corrections and unresolved assignments.':' Confirm orientation and numbering, then accept or reject each candidate.';
 $('showSetup').disabled=done;$('setupPanel').classList.toggle('hidden',done||$('setupPanel').classList.contains('hidden'));
 $('orientation').value=r.orientation;$('midline').value=r.guides.midline*100;$('archY').value=r.guides.arch_y*100;updateGuideValues();
 $('confirmOrientation').checked=r.orientation_confirmed;$('confirmNumbering').checked=r.numbering_confirmed;
 $('confirmOrientation').disabled=done;$('confirmNumbering').disabled=done;$('finalizeButton').disabled=done;$('finalizeButton').innerHTML=done?'✓ Review complete':'Complete review <span>→</span>';
 $('imageDimensions').textContent=`${r.image.width} × ${r.image.height}`;
 $('orientationMarker').textContent=r.orientation==='unknown'?'?':r.orientation==='flipped'?'L':'R';
 $('modelTiming').textContent=`${Math.round(r.model_runs.reduce((n,m)=>n+m.predict_ms,0))} ms · vision`;
 document.querySelectorAll('.filter').forEach(b=>b.classList.toggle('active',b.dataset.filter===state.filter));
 renderImage();renderOdontogram();renderFindings();renderInspector();renderReport();renderGraph();renderTrace();
}
function renderImage(){
 const r=state.case.report;const {width:w,height:h}=r.image;const showTeeth=$('teethLayer').checked,showFindings=$('findingsLayer').checked;
 let svg=`<image href="${esc(apiPath(`/api/cases/${state.case.id}/image`))}" x="0" y="0" width="${w}" height="${h}"/>`;
 if(showTeeth)for(const t of r.teeth){
  const [x,y,x2,y2]=t.bbox,selected=state.selected?.id===t.id;
  const color=t.flags.includes('duplicate_fdi')?'#e8988a':'#8bccb4';
  svg+=`<g data-region="${esc(t.id)}" data-type="tooth" tabindex="0" role="button" aria-label="Tooth ${t.fdi??t.id}"><rect x="${x}" y="${y}" width="${x2-x}" height="${y2-y}" rx="4" fill="${selected?'#68cba830':'transparent'}" stroke="${color}" stroke-width="${selected?3:1}" stroke-opacity="${selected?1:.52}" vector-effect="non-scaling-stroke"/><text x="${x+3}" y="${Math.max(15,y-6)}" fill="${color}" font-family="sans-serif" font-size="${w/95}" paint-order="stroke" stroke="#162329" stroke-width="3">${t.fdi??'?'}</text></g>`;
 }
 if(showFindings)for(const f of r.findings){
  if(f.review==='rejected')continue;const [x,y,x2,y2]=f.bbox,selected=state.selected?.id===f.id;
  svg+=`<g data-region="${esc(f.id)}" data-type="finding" tabindex="0" role="button" aria-label="${esc(f.title)}"><rect x="${x}" y="${y}" width="${x2-x}" height="${y2-y}" rx="4" fill="${selected?'#dcb26e25':'transparent'}" stroke="#dfb779" stroke-width="${selected?3:1.5}" stroke-dasharray="${selected?'none':'5 3'}" vector-effect="non-scaling-stroke"/><circle cx="${x2}" cy="${y}" r="${w/160}" fill="#d3ab70"/><text x="${x2}" y="${y+w/500}" text-anchor="middle" dominant-baseline="middle" fill="#18221e" font-size="${w/150}" font-family="sans-serif">${r.findings.indexOf(f)+1}</text></g>`;
 }
 if($('guidesLayer').checked){const x=Number($('midline').value)/100*w,y=Number($('archY').value)/100*h;svg+=`<path d="M${x} 0V${h}M0 ${y}H${w}" stroke="#bcd5d1" stroke-width="1" stroke-dasharray="7 7" vector-effect="non-scaling-stroke" opacity=".7"/>`;}
 $('imageSvg').innerHTML=svg;updateViewBox();
}
function toothShape(n){const type=n%10;return type>=6?'<path d="M5 4Q8 1 12 4Q16 1 19 4Q24 10 20 19L19 28Q17 34 15 27L12 21L9 28Q6 34 5 27L4 18Q0 9 5 4Z"/>':'<path d="M5 4Q12 0 19 4Q23 10 19 17L14 30Q12 35 10 30L5 17Q1 10 5 4Z"/>';}
function renderOdontogram(){
 const r=state.case.report,selected=state.selected?.type==='tooth'?r.teeth.find(t=>t.id===state.selected.id)?.fdi:r.findings.find(f=>f.id===state.selected?.id)?.fdi;
 const rows=[[18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28],[48,47,46,45,44,43,42,41,31,32,33,34,35,36,37,38]];
 $('odontogram').innerHTML=rows.map((row,i)=>`<div class="arch-row ${i?'lower':''}">${row.map(n=>{const slot=r.odontogram[n];return `<button class="tooth-slot ${slot.status==='detected'?'detected':''} ${slot.status==='conflict'?'conflict':''} ${slot.finding_ids.length?'finding':''} ${selected===n?'selected':''}" data-fdi="${n}" title="FDI ${n}: ${slot.status.replace('_',' ')}" aria-label="FDI ${n}: ${slot.status.replace('_',' ')}"><svg viewBox="0 0 24 36">${toothShape(n)}</svg><span>${n}</span></button>`;}).join('')}</div>`).join('');
 const unassigned=r.teeth.filter(t=>t.fdi===null).length,conflicts=r.teeth.filter(t=>t.flags.includes('duplicate_fdi')).length;
 $('assignmentQueue').innerHTML=r.teeth.filter(t=>t.fdi===null||t.flags.includes('duplicate_fdi')).map(t=>`<button data-tooth="${t.id}">${t.fdi?'Conflict: '+t.fdi:'Unassigned'} · ${t.id}</button>`).join('');
 $('unassignedCount').textContent=`${unassigned} unassigned${conflicts?` · ${conflicts} conflicting assignments`:''}`;
}
function renderFindings(){
 const r=state.case.report,findings=r.findings.filter(f=>state.filter==='all'||f.category===state.filter);
 $('findingCount').textContent=r.findings.length;$('reviewProgress').textContent=`${r.findings.filter(f=>f.review!=='unreviewed').length}/${r.findings.length} reviewed`;
 $('findingList').innerHTML=findings.length?findings.map(f=>`<button class="finding-item ${esc(f.category)} ${esc(f.review)} ${state.selected?.id===f.id?'selected':''}" data-finding="${esc(f.id)}"><span class="finding-symbol">${f.category==='pathology'?'◇':f.category==='restoration'?'▧':'⌁'}</span><span class="finding-copy"><strong>${esc(f.title)}</strong><small>${f.fdi?`Tooth ${f.fdi}`:f.association==='region'?'Region finding':'Tooth unresolved'} · ${f.review==='unreviewed'?'Needs review':f.review}</small></span><span class="finding-score">${f.review==='accepted'?'✓':f.review==='rejected'?'×':Math.round(f.score*100)+'%'}</span></button>`).join(''):'<div class="empty-findings">No candidates in this view. Non-detection does not establish absence of disease.</div>';
}
function select(type,id){state.selected={type,id};renderImage();renderOdontogram();renderFindings();renderInspector();renderGraph();}
function fdiOptions(value){return `<option value="">Unassigned</option>`+FDI.map(n=>`<option value="${n}" ${value===n?'selected':''}>FDI ${n}</option>`).join('');}
function renderInspector(){
 if(!state.selected){$('inspector').innerHTML='<div class="empty-inspector"><span>⌖</span><h3>Look a little closer</h3><p>Select a finding or tooth to inspect its image region, evidence, and assignment.</p></div>';return;}
 const r=state.case.report,done=r.status==='reviewed';
 const item=(state.selected.type==='tooth'?r.teeth:r.findings).find(t=>t.id===state.selected.id);if(!item){state.selected=null;renderInspector();return;}
 const crop=apiPath(`/api/cases/${state.case.id}/crop/${item.id}`);
 if(state.selected.type==='tooth'){
  $('inspector').innerHTML=`<div class="inspector-title"><h3>${item.fdi?'Tooth '+item.fdi:'Unassigned tooth'}</h3><span>${esc(item.id)}</span></div><img class="crop-image" src="${esc(crop)}" alt="Context crop for selected tooth"><div class="inspection-meta"><span>Predicted tooth type ${item.tooth_type}</span><span>Score ${Math.round(item.score*100)}%</span></div><label for="toothFdi">FDI assignment</label><select id="toothFdi" ${done?'disabled':''}>${fdiOptions(item.fdi)}</select><button class="button secondary full" id="saveTooth" style="margin-top:10px" ${done?'disabled':''}>Save tooth assignment</button>${item.flags.length?`<p class="warning-inline">${item.flags.map(esc).join(' · ').replaceAll('_',' ')}</p>`:''}<p class="vlm-note">Tooth type comes from the model. Quadrant assignment uses image geometry and requires review. Unassigned is a valid outcome.</p><div class="tooth-list">${r.teeth.filter(t=>t.fdi===item.fdi&&t.id!==item.id&&item.fdi!==null).map(t=>`<button data-tooth="${t.id}">Also assigned here: ${t.id}</button>`).join('')}</div>`;
  $('saveTooth').onclick=()=>saveReview({tooth_assignments:{[item.id]:$('toothFdi').value?Number($('toothFdi').value):null}});return;
 }
 const refs=r.references.filter(ref=>item.reference_ids.includes(ref.id));
 $('inspector').innerHTML=`<div class="inspector-title"><h3>${esc(item.title)}</h3><span>${Math.round(item.score*100)}% score</span></div><img class="crop-image" src="${esc(crop)}" alt="Context crop for ${esc(item.title)}"><div class="inspection-meta"><span>${esc(item.model)} · ${esc(item.raw_label)}</span></div><label for="findingTooth">Associated tooth</label><select id="findingTooth" ${done?'disabled':''}><option value="">Unassigned / region</option>${r.teeth.map(t=>`<option value="${t.id}" ${item.tooth_id===t.id?'selected':''}>${t.fdi?'FDI '+t.fdi:'Unassigned'} · ${t.id} · type ${t.tooth_type}</option>`).join('')}</select><p class="vlm-note">Association: ${esc(item.association)}. Model scores are not disease probabilities.</p><label for="findingNotes">Review note</label><textarea id="findingNotes" maxlength="2000" placeholder="Optional observation…" ${done?'disabled':''}>${esc(item.notes)}</textarea><button class="button secondary full" id="saveFinding" style="margin-top:8px" ${done?'disabled':''}>Save association & note</button><div class="review-actions"><button class="accept-button ${item.review==='accepted'?'chosen':''}" id="acceptFinding" ${done?'disabled':''}>✓ Accept candidate</button><button class="reject-button ${item.review==='rejected'?'chosen':''}" id="rejectFinding" ${done?'disabled':''}>× Reject</button></div>${refs.map(ref=>`<div class="source-snippet"><small>REFERENCE CONTEXT · ${esc(ref.publisher)}</small><p>${esc(ref.consideration)}</p><a href="${esc(ref.url)}" target="_blank" rel="noopener noreferrer">Read source ↗</a><p class="vlm-note">Curated summary · ${esc(ref.section)}</p></div>`).join('')}${!refs.length?'<p class="vlm-note" style="margin-top:15px">No applicable reference in the starter corpus for this finding. No treatment recommendation generated.</p>':''}${item.vlm?renderVLM(item.vlm):`<button class="vlm-button" id="inspectCrop" ${done||!state.health?.gemini_configured?'disabled':''}>✧ Inspect crop with Gemini</button><p class="vlm-note">Optional. Sends this crop to Google. Returns an unverified description; never changes your decision.</p>`}`;
 const patch=()=>({finding_teeth:{[item.id]:$('findingTooth').value||null},notes:{[item.id]:$('findingNotes').value}});
 $('saveFinding').onclick=()=>saveReview(patch());
 $('acceptFinding').onclick=()=>saveReview({...patch(),finding_reviews:{[item.id]:'accepted'}});
 $('rejectFinding').onclick=()=>saveReview({...patch(),finding_reviews:{[item.id]:'rejected'}});
 if($('inspectCrop'))$('inspectCrop').onclick=async()=>{
  if(!confirm('Send this selected image crop to Google Gemini for an unverified visual description? Only use images you are authorized to share.'))return;
  const button=$('inspectCrop');button.disabled=true;button.textContent='Inspecting crop…';
  try{state.case=await api(`/api/cases/${state.case.id}/inspect/${item.id}`,jsonOptions('POST',{consent:true}));renderCase();toast('Crop observation added. Your review decision is unchanged.');}
  catch(e){toast(e.message,true);state.case=await api(`/api/cases/${state.case.id}`);renderCase();}
 };
}
function renderVLM(v){return `<div class="vlm-result"><strong>Unverified visual description</strong><ul>${v.visible_features.map(s=>`<li>${esc(s)}</li>`).join('')}</ul><strong>Limitations</strong><ul>${v.limitations.map(s=>`<li>${esc(s)}</li>`).join('')}</ul><p>${esc(v.review_question)}</p><span class="vlm-note">${esc(v.model)} · Separate from the reviewer decision</span></div>`;}
async function saveReview(patch){
 if(state.busy)return;state.busy=true;
 try{state.case=await api(`/api/cases/${state.case.id}/review`,jsonOptions('PATCH',{expected_revision:state.case.report.revision,...patch}));renderCase();toast('Review saved.');}
 catch(e){toast(e.message,true);try{state.case=await api(`/api/cases/${state.case.id}`);renderCase();}catch{}}
 finally{state.busy=false;}
}
function renderReport(){
 const r=state.case.report,active=r.findings.filter(f=>f.review!=='rejected');
 $('reportTab').innerHTML=`<div class="report-summary"><div class="report-stat"><strong>${r.teeth.length}</strong><span>TEETH DETECTED</span></div><div class="report-stat"><strong>${r.findings.filter(f=>f.review==='accepted').length}</strong><span>ACCEPTED CANDIDATES</span></div><div class="report-stat"><strong>${r.findings.filter(f=>f.review==='unreviewed').length}</strong><span>AWAITING REVIEW</span></div></div>${active.map(f=>`<div class="report-line"><span class="fdi-tag">${f.fdi?'FDI '+f.fdi:'REGION / ?'}</span><div><strong style="font-weight:500">${esc(f.title)}</strong>${f.notes?`<p>${esc(f.notes)}</p>`:''}</div><span class="state">${esc(f.review)}</span></div>`).join('')||'<p class="report-note">No active finding candidates. This is not a normal-examination result.</p>'}${[...new Map(r.considerations.map(c=>[c.reference_id,c])).values()].map(c=>{const ref=r.references.find(x=>x.id===c.reference_id);return `<div class="report-note">${esc(c.text)} ${ref?`<a href="${esc(ref.url)}" target="_blank" rel="noopener noreferrer">[${esc(ref.publisher)} ↗]</a>`:''}</div>`;}).join('')}<div class="report-note"><strong>Assessment limits</strong><br>${r.limitations.map(esc).join('<br>')}</div>`;
}
function renderGraph(){
 const r=state.case.report,g=r.evidence_graph;
 $('evidenceTab').innerHTML=`<p class="graph-summary">Every finding retains its image region, model provenance, association, and reference context. Reference links contextualize an observation; they do not confirm it.</p><div class="graph-links"><div class="graph-node">IMAGE<b>1 panoramic radiograph</b></div><div class="graph-node">VISION<b>${r.model_runs.length} recorded model runs</b></div><div class="graph-node">ANATOMY<b>${r.teeth.length} tooth candidates</b></div><div class="graph-node">OBSERVATIONS<b>${r.findings.length} findings</b></div><div class="graph-node">SOURCES<b>${r.references.length} applicable references</b></div></div><div style="margin-top:15px">${g.edges.filter(e=>e.source===state.selected?.id||(!state.selected&&e.source.includes('-f'))).slice(0,15).map(e=>`<div class="graph-edge"><span>${esc(e.source)}</span> → ${esc(e.type.replaceAll('_',' '))} → <span>${esc(e.target)}</span></div>`).join('')}</div><p class="vlm-note">${g.nodes.length} nodes · ${g.edges.length} edges · Full graph included in JSON export.</p>`;
}
function renderTrace(){const r=state.case.report;$('activityTab').innerHTML=r.trace.map(e=>`<div class="timeline-entry"><i></i><div><strong>${esc(e.node.replaceAll('_',' '))}</strong><p>${esc(e.detail)}</p></div><time>${new Date(e.at).toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit',second:'2-digit'})}</time></div>`).join('');}
function updateGuideValues(){$('midlineValue').textContent=`${Number($('midline').value).toFixed(1)}%`;$('archValue').textContent=`${Number($('archY').value).toFixed(1)}%`;}
async function beginUpload(file){
 if(!file)return;if(file.size>20*1024*1024){toast('Choose an image under 20 MB.',true);return;}
 setView('processing');$('processStage').textContent='Uploading panoramic image';
 const data=new FormData();data.append('image',file);data.append('orientation','unknown');data.append('sensitivity',$('sensitivityMode').checked?'true':'false');
 try{state.case=await api('/api/cases',{method:'POST',body:data});state.selected=null;resetZoom();renderCase();refreshCases();}
 catch(e){setView('welcome');toast(e.message,true);}
 $('fileInput').value='';
}
function renderReferences(records){$('referenceResults').innerHTML=records.map(r=>`<article class="reference-card"><small>${esc(r.publisher.toUpperCase())}</small><h3>${esc(r.title)}</h3><p>${esc(r.text)}</p><a href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">Open original source ↗</a><footer>${esc(r.kind.replaceAll('_',' '))} · ${esc(r.section)}<br>Reviewed ${esc(r.reviewed_at)}${r.retrieval_score!==undefined?' · retrieval score '+r.retrieval_score:''}</footer></article>`).join('');}
async function showLibrary(){setView('library');$('breadcrumbCase').textContent='Reference library';try{renderReferences(await api('/api/references'));}catch(e){toast(e.message,true);}}
$('demoButton').onclick=async()=>{setView('processing');$('processStage').textContent='Preparing research example';try{state.case=await api('/api/demo',jsonOptions('POST',{sensitivity:$('sensitivityMode').checked}));state.selected=null;resetZoom();renderCase();refreshCases();}catch(e){setView('welcome');toast(e.message,true);}};
$('dropzone').onclick=()=>$('fileInput').click();$('dropzone').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('fileInput').click();}};
$('fileInput').onchange=e=>beginUpload(e.target.files[0]);
for(const event of ['dragenter','dragover'])$('dropzone').addEventListener(event,e=>{e.preventDefault();$('dropzone').classList.add('dragover');});
for(const event of ['dragleave','drop'])$('dropzone').addEventListener(event,e=>{e.preventDefault();$('dropzone').classList.remove('dragover');});
$('dropzone').addEventListener('drop',e=>beginUpload(e.dataTransfer.files[0]));
$('newCase').onclick=()=>{clearTimeout(state.poll);setView('welcome');$('breadcrumbCase').textContent='New analysis';};
$('workspaceNav').onclick=()=>state.case?renderCase():setView('welcome');$('referencesNav').onclick=showLibrary;$('auditNav').onclick=showAudit;$('refreshCases').onclick=refreshCases;
$('caseList').onclick=e=>{const b=e.target.closest('[data-case]');if(b)openCase(b.dataset.case);};
$('findingList').onclick=e=>{const b=e.target.closest('[data-finding]');if(b)select('finding',b.dataset.finding);};
$('inspector').addEventListener('click',e=>{const b=e.target.closest('[data-tooth]');if(b)select('tooth',b.dataset.tooth);});
$('assignmentQueue').onclick=e=>{const b=e.target.closest('[data-tooth]');if(b)select('tooth',b.dataset.tooth);};
$('odontogram').onclick=e=>{const b=e.target.closest('[data-fdi]');if(!b)return;const r=state.case.report,slot=r.odontogram[b.dataset.fdi];if(slot.finding_ids.length)select('finding',slot.finding_ids[0]);else if(slot.tooth_ids.length)select('tooth',slot.tooth_ids[0]);else toast(`FDI ${b.dataset.fdi} was not detected. This does not confirm that the tooth is missing.`);};
for(const b of document.querySelectorAll('.filter'))b.onclick=()=>{state.filter=b.dataset.filter;document.querySelectorAll('.filter').forEach(x=>x.classList.toggle('active',x===b));renderFindings();};
for(const b of document.querySelectorAll('.tab'))b.onclick=()=>{state.tab=b.dataset.tab;document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===b));for(const tab of ['report','evidence','activity'])$(tab+'Tab').classList.toggle('hidden',tab!==state.tab);};
for(const id of ['teethLayer','findingsLayer','guidesLayer'])$(id).onchange=renderImage;
$('zoomIn').onclick=()=>{state.zoom=Math.min(5,state.zoom*1.25);updateViewBox();};$('zoomOut').onclick=()=>{state.zoom=Math.max(1,state.zoom/1.25);if(state.zoom===1){state.panX=state.panY=0;}updateViewBox();};$('resetView').onclick=resetZoom;
let drag=null,moved=false;
$('imageSvg').addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,px:state.panX,py:state.panY};moved=false;});
$('imageSvg').addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>4)moved=true;if(!moved)return;const scale=state.case.report.image.width/($('imageSvg').clientWidth*state.zoom);state.panX=drag.px-dx*scale;state.panY=drag.py-dy*scale;updateViewBox();});
window.addEventListener('pointerup',()=>{drag=null;});
$('imageSvg').addEventListener('click',e=>{if(moved)return;const region=e.target.closest('[data-region]');if(region)select(region.dataset.type,region.dataset.region);});
$('imageSvg').addEventListener('keydown',e=>{if(e.key==='Enter'){const region=e.target.closest('[data-region]');if(region)select(region.dataset.type,region.dataset.region);}});
$('imageSvg').addEventListener('wheel',e=>{e.preventDefault();state.zoom=Math.min(5,Math.max(1,state.zoom*(e.deltaY<0?1.12:1/1.12)));if(state.zoom===1)state.panX=state.panY=0;updateViewBox();},{passive:false});
$('showSetup').onclick=()=>{$('setupPanel').classList.toggle('hidden');$('guidesLayer').checked=!$('setupPanel').classList.contains('hidden');renderImage();};
for(const id of ['midline','archY'])$(id).oninput=()=>{updateGuideValues();$('guidesLayer').checked=true;renderImage();};
$('saveGuides').onclick=()=>saveReview({orientation:$('orientation').value,midline:Number($('midline').value)/100,arch_y:Number($('archY').value)/100});
$('confirmOrientation').onchange=()=>saveReview({orientation_confirmed:$('confirmOrientation').checked});$('confirmNumbering').onchange=()=>saveReview({numbering_confirmed:$('confirmNumbering').checked});
$('finalizeButton').onclick=async()=>{try{state.case=await api(`/api/cases/${state.case.id}/finalize`,{method:'POST'});renderCase();refreshCases();toast('Review completed. The evidence record is ready to export.');}catch(e){toast(e.message,true);}};
$('referenceSearch').onsubmit=async e=>{e.preventDefault();const b=e.target.querySelector('button');b.disabled=true;try{renderReferences(await api('/api/references/search?q='+encodeURIComponent($('referenceQuery').value)));}catch(error){toast(error.message,true);}finally{b.disabled=false;}};
(async()=>{try{state.health=await api('/api/health');$('deviceLabel').textContent=`${state.health.device} · ready`;}catch{$('deviceLabel').textContent='Service unavailable';}await refreshCases();})();

$('deleteCase').onclick=async()=>{if(!confirm('Delete this case, its image, report and workflow history? This cannot be undone.'))return;try{await api(`/api/cases/${state.case.id}`,{method:'DELETE'});state.case=null;state.selected=null;setView('welcome');$('breadcrumbCase').textContent='Overview';refreshCases();toast('Case deleted.');}catch(e){toast(e.message,true);}};

async function showAudit(){
 setView('audit');$('breadcrumbCase').textContent='Model audit';
 try{
  const d=await api('/api/evaluation');const tooth=d.tooth;
  const accuracy=(x)=>x===null?'—':`${Math.round(x*100)}%`;
  let html=`<div class="audit-intro">These figures test specific model behaviors on 50 DENTEX validation images. Annotations cover <strong>abnormal teeth only</strong>; they cannot measure detection of every healthy tooth. The lesion metric checks whether a predicted box center lands inside a labeled tooth site. It is not mAP or clinical sensitivity.</div>`;
  html+=`<div class="audit-highlights"><div><strong>${tooth.annotated_abnormal_teeth}</strong><span>ANNOTATED ABNORMAL TEETH</span></div><div><strong>${tooth.localized_iou_0_3}</strong><span>TOOTH BOXES LOCALIZED · IoU ≥ 0.3</span></div><div><strong>${tooth.exact_fdi_after_localization}</strong><span>EXACT FDI AFTER LOCALIZATION</span></div></div>`;
  for(const [category,label] of [['suspected_caries','Caries + deep caries'],['periapical_radiolucency','Periapical lesion sites'],['impaction_candidate','Impacted tooth sites']]){
   html+=`<div class="audit-card"><div class="audit-card-header"><div><small>FINDING SITE AUDIT</small><h3>${label}</h3></div><span>Published checkpoint thresholds</span></div>`;
   for(const [model,title] of [['yolo26','YOLO26 · primary'],['liodon','Liodon · optional']]){
    const m=d.findings[model]?.[category];if(!m)continue;
    html+=`<div class="audit-row"><strong>${title}</strong><div class="audit-bars"><div><span>Site recall</span><div class="track"><i style="width:${m.recall*100}%"></i></div><b>${accuracy(m.recall)}</b></div><div><span>Site precision</span><div class="track muted-track"><i style="width:${m.precision*100}%"></i></div><b>${accuracy(m.precision)}</b></div></div><small>${m.tp} matched · ${m.fn} missed · ${m.fp} unmatched</small></div>`;
   }
   html+='</div>';
  }
  html+=`<div class="audit-caveat"><strong>How to read this</strong><p>${esc(d.metrics_definition.limitations)} DENTEX uses whole-tooth abnormality boxes while models may predict smaller lesion boxes. Caries and deep caries are merged. The primary model’s stronger precision at its published threshold comes with low caries site recall on this split; optional sensitivity mode exposes additional Liodon leads for human review. Liodon was already selected on DENTEX validation, so these numbers are not independent confirmation.</p><a href="${esc(d.dataset_url)}" target="_blank" rel="noopener noreferrer">Open DENTEX dataset ↗</a><br><a href="${esc(apiPath('/api/evaluation'))}" target="_blank" rel="noopener noreferrer">View machine-readable evaluation ↗</a></div>`;
  $('auditContent').innerHTML=html;
 }catch(e){$('auditContent').textContent=e.message;toast(e.message,true);}
}

$('signOut').onclick=async()=>{await fetch('/api/logout',{method:'POST'});location.replace('/login');};
