import json
import pytest
from PIL import Image
from domain import Finding
from vlm import inspect_crop


class MockClient:
    status=200
    content={'candidates':[{'content':{'parts':[{'text':json.dumps({
        'visible_features':['Synthetic region'], 'limitations':['Not a diagnostic image'],
        'review_question':'Is an original image available?'})}]}}]}
    def __init__(self,**kwargs):pass
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def post(self,url,headers,json):
        assert 'key=' not in url
        assert json['generationConfig']['responseJsonSchema']
        self.status_code=self.status
        return self
    def json(self):return self.content


def test_vlm_validates_output_and_handles_quota(tmp_path,monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','test-key')
    monkeypatch.setattr('vlm.httpx.Client',MockClient)
    image=tmp_path/'test.png';Image.new('RGB',(100,100)).save(image)
    f=Finding(id='f',label='filling',title='Filling',category='restoration',raw_label='filling',model='test',bbox=[0,0,50,50],score=.8)
    result=inspect_crop(image,f)
    assert result['status']=='unverified_observation'
    monkeypatch.setattr(MockClient,'status',429)
    with pytest.raises(RuntimeError,match='quota'):
        inspect_crop(image,f)
