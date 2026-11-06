"""The real corpus API opts only calibrated, unbaked motions into jaw candidates."""
import json
import sqlite3

from test_animation_corpus import corpus_root, make_client, SAMPLE, PREFIX, encode


TEMPLATE = '376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992'
AVATAR = 'b42c21dacb5d07617790c40c69f68a074aca58d74d76b898f2cb98bbbf96862c'


def test_calibrated_corpus_exposes_unreviewed_jaw_protocol_without_baked_export(corpus_root,tmp_path):
    with sqlite3.connect(corpus_root/'corpus.sqlite3') as db:
        for name,key in [('build','template_sha256'),('skeleton','smplx_rest_model_sha256')]:
            value=json.loads(db.execute('SELECT value FROM metadata WHERE key=?',(name,)).fetchone()[0])
            value[key]=TEMPLATE
            db.execute('UPDATE metadata SET value=? WHERE key=?',(encode(value),name))
    client,_=make_client(corpus_root,tmp_path)
    with client:
        clip=client.get(PREFIX+'/motions/'+SAMPLE).json()['clip']
        profile=clip.get('nonmanual_candidate')
        assert profile is not None
        assert profile == {'profile':'smplx-jaw-godette-calibrated-v1','source_joint':22,
                           'source_template_sha256':TEMPLATE,'target_avatar_sha256':AVATAR,
                           'status':'research_candidate','export_supported':False}
        assert clip['poses'][0][22] == [0,0,0]
        assert clip['expression_review']=='unverified'
        assert 'expression' in clip['unmapped_target_channels']
        assert client.get('/animation/static/nonmanual.mjs').status_code==200
        old=client.get('/v1/animation/motions/hello').json()
        assert 'nonmanual_candidate' not in old['clip']


def test_other_templates_do_not_silently_gain_the_calibration(corpus_root,tmp_path):
    client,_=make_client(corpus_root,tmp_path)
    with client:
        clip=client.get(PREFIX+'/motions/'+SAMPLE).json()['clip']
        assert 'nonmanual_candidate' not in clip
        assert 'jaw' in clip['unmapped_target_channels']
