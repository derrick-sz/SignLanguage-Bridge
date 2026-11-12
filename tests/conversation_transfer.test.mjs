import test from 'node:test';
import assert from 'node:assert/strict';
import {createRealtimeTransfer,validateRealtimeTransfer,RealtimeHandoffSender,RealtimeHandoffReceiver,TRANSFER_PROTOCOL} from '../slbridge/animation/web/communication/conversation-transfer.mjs';
import {Conversation,readableTranscript} from '../slbridge/animation/web/communication/conversation.mjs';

const caption=()=>({id:'session:0',status:'committed',revision:1,confirmed:true,text:'核对后的文字。',original_text:'机器原译文。',
  word_evidence:['原词'],review_hint:{reason:'single_word_expansion'},source:{kind:'training_replay',label:'训练示范回放 · CSL-Daily train',id:'school-reading'}});
const origin='http://127.0.0.1:8879';
const until=async predicate=>{for(let i=0;i<100&&!predicate();i++)await new Promise(r=>setTimeout(r,1));assert.ok(predicate());};

function pair({automatic=true,blocked=false}={}){
  const trace=[],timers=new Map(),states=[],conversation=new Conversation();let next=0,time=0,opens=0;
  let sender,receiver;
  const opener={postMessage(data,target){trace.push({to:'opener',data,target});if(automatic)queueMicrotask(()=>sender.receive({data,origin,source:child}));}};
  const child={closed:false,focus(){},postMessage(data,target){trace.push({to:'child',data,target});if(automatic)queueMicrotask(()=>receiver.receive({data,origin,source:opener}));}};
  sender=new RealtimeHandoffSender({origin,openWindow:()=>{opens++;return blocked?null:child;},uid:()=>`nonce-${++next}-abcdefgh`,
    now:()=>time,setTimer:(fn,ms)=>{timers.set(++next,{fn,due:time+ms});return next;},clearTimer:id=>timers.delete(id),listen:()=>()=>{},onState:s=>states.push(s)});
  receiver=new RealtimeHandoffReceiver({origin,opener,now:()=>time,listen:()=>()=>{},onTransfer:payload=>conversation.confirmRealtime(payload)});
  return {sender,receiver,child,opener,trace,states,conversation,get opens(){return opens;},
    setAutomatic(value){automatic=value;},
    tick(value){time=value;const due=[...timers].filter(([,job])=>job.due<=time);for(const [id] of due)timers.delete(id);for(const [,job] of due)job.fn();}};
}

test('confirmed captions have stable content IDs and keep bounded raw evidence without target annotations',async()=>{
  const source={...caption(),annotation:'不应进入对话的练习目标'};
  const a=await createRealtimeTransfer(source),b=await createRealtimeTransfer(source);
  assert.equal(a.id,b.id);assert.equal(a.annotation,undefined);assert.equal(a.original_text,'机器原译文。');
  const checked=await validateRealtimeTransfer(a);assert.deepEqual(checked.word_evidence,['原词']);
  assert.equal(checked.source.kind,'training_replay');
  assert.notEqual((await createRealtimeTransfer({...source,text:'再次人工修订。'})).id,a.id);
  assert.equal(source.text,'核对后的文字。');
});

test('unconfirmed, empty, incompatible and oversized payloads are refused',async()=>{
  for(const change of [{confirmed:false},{status:'pending'},{text:' '},{text:'长'.repeat(8001)},{revision:0}])
    await assert.rejects(createRealtimeTransfer({...caption(),...change}));
  const good=await createRealtimeTransfer(caption());
  for(const change of [{text:'被篡改'},{id:'rt_'+'0'.repeat(64)},{extra:'not allowed'},{source:{kind:'test',label:'伪标签'}}])
    await assert.rejects(validateRealtimeTransfer({...good,...change}));
});

test('conversation accepts only locally validated transfers and keeps confirmation distinct from model review',async()=>{
  const c=new Conversation(),wire=await createRealtimeTransfer(caption());
  assert.throws(()=>c.confirmRealtime(wire));
  const checked=await validateRealtimeTransfer(wire);
  assert.equal(c.confirmRealtime(checked),true);assert.equal(c.confirmRealtime(checked),false);
  const row=c.messages[0];assert.equal(row.text_confirmed_by_user,true);assert.equal(row.review_status,'unverified');
  assert.equal(row.original_text,'机器原译文。');assert.equal(row.provenance.source.kind,'training_replay');
  c.addReply('你好。','reply');const text=readableTranscript(c.export());
  assert.match(text,/训练示范/);assert.match(text,/机器原译文/);assert.match(text,/你好/);
  assert.doesNotMatch(text,/model_generation|schema_version/);
});

test('the handshake imports once, reuses its dialogue window and preserves edited revisions',async()=>{
  const h=pair();await h.sender.send(caption());await h.sender.send(caption());
  assert.equal(h.opens,1);assert.equal(h.conversation.messages.length,1);
  await h.sender.send({...caption(),text:'人工新修订。'});
  assert.equal(h.conversation.messages.length,2);assert.equal(h.opens,1);
  assert.ok(h.trace.every(entry=>entry.target===origin));h.sender.dispose();h.receiver.dispose();
});

test('a timed-out transfer preserves the open dialogue window and its draft for retry',async()=>{
  const h=pair();await h.sender.send(caption());h.child.replyDraft='尚未发送的回复';h.setAutomatic(false);
  const count=h.trace.length,pending=h.sender.send({...caption(),text:'第二句'});
  const rejected=assert.rejects(pending,/超时/);
  await until(()=>h.trace.slice(count).some(x=>x.data.kind==='hello'));h.tick(10001);await rejected;
  assert.equal(h.sender.target,h.child);assert.equal(h.child.closed,false);
  h.setAutomatic(true);await h.sender.send({...caption(),text:'重试后的句子'});
  assert.equal(h.opens,1);assert.equal(h.child.replyDraft,'尚未发送的回复');
  assert.equal(h.conversation.messages.length,2);h.sender.dispose();h.receiver.dispose();
});

test('wrong origin, window or nonce cannot make an unacknowledged transfer succeed',async()=>{
  const h=pair({automatic:false}),pending=h.sender.send(caption());const rejected=assert.rejects(pending,/响应|超时/);
  await until(()=>h.trace.some(x=>x.data.kind==='hello'));
  const {nonce,expiresAt}=h.trace.find(x=>x.data.kind==='hello').data;
  for(const event of [{origin:'https://elsewhere.test',source:h.child,nonce},{origin,source:{},nonce},
    {origin,source:h.child,nonce:'other-nonce'}])h.sender.receive({...event,data:{protocol:TRANSFER_PROTOCOL,kind:'ready',nonce:event.nonce,expiresAt}});
  assert.equal(h.trace.some(x=>x.data.kind==='transfer'),false);h.tick(10001);await rejected;h.sender.dispose();h.receiver.dispose();
});

test('an acknowledgement requires a sent payload and the matching content ID',async()=>{
  const h=pair({automatic:false}),payload=await createRealtimeTransfer(caption()),pending=h.sender.send(caption());
  await until(()=>h.trace.some(x=>x.data.kind==='hello'));
  const envelope=h.trace.find(x=>x.data.kind==='hello').data;
  const send=data=>h.sender.receive({origin,source:h.child,data:{...envelope,...data}});
  send({kind:'accepted',id:payload.id,added:true});assert.equal(h.sender.busy,true);
  send({kind:'ready'});assert.equal(h.trace.at(-1).data.kind,'transfer');
  send({kind:'accepted',id:'rt_'+'0'.repeat(64),added:true});assert.equal(h.sender.busy,true);
  send({kind:'accepted',id:payload.id,added:true});assert.deepEqual(await pending,{added:true});
  h.sender.dispose();h.receiver.dispose();
});

test('receiver refuses unrelated peers and expired, cancelled or disposed transfers',async()=>{
  const payload=await createRealtimeTransfer(caption());
  for(const finish of ['expire','cancel','dispose']){
    const h=pair({automatic:false}),data={protocol:TRANSFER_PROTOCOL,kind:'hello',nonce:'nonce-abcdefgh',expiresAt:10000};
    for(const event of [{origin:'https://elsewhere.test',source:h.opener},{origin,source:{} }])
      await h.receiver.receive({...event,data});
    assert.equal(h.trace.length,0);
    await h.receiver.receive({origin,source:h.opener,data:{...data,expiresAt:10001}});assert.equal(h.trace.length,0);
    await h.receiver.receive({origin,source:h.opener,data});assert.equal(h.trace.at(-1).data.kind,'ready');
    const pending=h.receiver.receive({origin,source:h.opener,data:{...data,kind:'transfer',payload}});
    if(finish==='expire')h.tick(10000);
    if(finish==='cancel')await h.receiver.receive({origin,source:h.opener,data:{...data,kind:'cancel'}});
    if(finish==='dispose')h.receiver.dispose();
    await pending;assert.equal(h.conversation.messages.length,0);assert.equal(h.trace.some(x=>x.data.kind==='accepted'),false);
    h.sender.dispose();h.receiver.dispose();
  }
});

test('UTF-8 size bounds and immutable validation also cover multibyte evidence',async()=>{
  await assert.rejects(createRealtimeTransfer({...caption(),text:'字'.repeat(8000),original_text:'字'.repeat(8000),
    word_evidence:Array(128).fill('词'.repeat(128)),review_hint:null}),/过长/);
  await assert.rejects(createRealtimeTransfer({...caption(),source:{kind:'camera',label:'摄像头',id:123}}));
  const checked=await validateRealtimeTransfer(await createRealtimeTransfer(caption()));
  assert.throws(()=>{checked.text='修改';});assert.throws(()=>{checked.source.label='改来源';});
  assert.throws(()=>{checked.word_evidence.push('加词');});
});

test('a blocked or closed popup has a recoverable failure',async()=>{
  const blocked=pair({blocked:true});await assert.rejects(blocked.sender.send(caption()),/窗口/);blocked.sender.dispose();blocked.receiver.dispose();
  const h=pair({automatic:false}),pending=h.sender.send(caption());const rejected=assert.rejects(pending,/关闭/);
  await until(()=>h.trace.length>0);h.child.closed=true;h.tick(250);await rejected;h.sender.dispose();h.receiver.dispose();
});

test('receiver validates peers and abandons an older nonce during asynchronous validation',async()=>{
  const h=pair({automatic:false}),payload=await createRealtimeTransfer(caption());
  await h.receiver.receive({origin:'https://elsewhere.test',source:h.opener,data:{protocol:TRANSFER_PROTOCOL,kind:'hello',nonce:'nonce-abcdefgh',expiresAt:10000}});
  assert.equal(h.trace.length,0);
  const base={origin,source:h.opener};await h.receiver.receive({...base,data:{protocol:TRANSFER_PROTOCOL,kind:'hello',nonce:'nonce-abcdefgh',expiresAt:10000}});
  assert.equal(h.trace[0].data.kind,'ready');
  const old=h.receiver.receive({...base,data:{protocol:TRANSFER_PROTOCOL,kind:'transfer',nonce:'nonce-abcdefgh',expiresAt:10000,payload}});
  await h.receiver.receive({...base,data:{protocol:TRANSFER_PROTOCOL,kind:'hello',nonce:'nonce-new-abcdefgh',expiresAt:10000}});await old;
  assert.equal(h.conversation.messages.length,0);h.sender.dispose();h.receiver.dispose();
});

test('leaving the realtime page cancels a pending handshake without a late acknowledgement',async()=>{
  const h=pair({automatic:false}),pending=h.sender.send(caption());const rejected=assert.rejects(pending,/取消/);
  await until(()=>h.trace.length>0);h.sender.dispose();await rejected;
  const nonce=h.trace[0].data.nonce;h.sender.receive({origin,source:h.child,data:{protocol:TRANSFER_PROTOCOL,kind:'accepted',nonce,id:'old'}});
  assert.notEqual(h.states.at(-1).status,'sent');h.receiver.dispose();
});
