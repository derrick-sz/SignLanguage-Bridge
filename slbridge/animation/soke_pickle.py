"""Read SOKE metadata/numeric pickles as data; never import/call pickle globals."""
from dataclasses import dataclass
import math
import pickletools

import numpy as np


@dataclass
class Opaque:
    kind: str
    args: object
    state: object = None


def parse_pickle(raw: bytes):
    if len(raw)>32*1024*1024:
        raise ValueError('Pickle exceeds metadata bound')
    stack=[];memo={};mark=object()
    def marked():
        values=[]
        while stack and stack[-1] is not mark:values.append(stack.pop())
        if not stack:raise ValueError('Missing pickle MARK')
        stack.pop();return values[::-1]
    result=None;stopped=False
    try:
        for count,(op,arg,pos) in enumerate(pickletools.genops(raw)):
            if count>1500000 or len(stack)>100000:raise ValueError('Pickle complexity bound exceeded')
            name=op.name
            if name in ('PROTO','FRAME'):continue
            if name=='STOP':
                if len(stack)!=1 or pos+1!=len(raw):raise ValueError('Unexpected pickle ending')
                result=stack[0];stopped=True;break
            if name=='MARK':stack.append(mark)
            elif name=='EMPTY_LIST':stack.append([])
            elif name=='EMPTY_DICT':stack.append({})
            elif name=='EMPTY_TUPLE':stack.append(())
            elif name=='NONE':stack.append(None)
            elif name=='NEWTRUE':stack.append(True)
            elif name=='NEWFALSE':stack.append(False)
            elif name in ('BINUNICODE','SHORT_BINUNICODE','BINUNICODE8','UNICODE','BINSTRING','SHORT_BINSTRING','BINBYTES','SHORT_BINBYTES','BINBYTES8','BININT','BININT1','BININT2','INT','LONG','LONG1','LONG4','BINFLOAT','FLOAT'):stack.append(arg)
            elif name in ('BINGET','LONG_BINGET','GET'):stack.append(memo[int(arg)])
            elif name=='MEMOIZE':memo[len(memo)]=stack[-1]
            elif name in ('BINPUT','LONG_BINPUT','PUT'):memo[int(arg)]=stack[-1]
            elif name=='TUPLE':stack.append(tuple(marked()))
            elif name in ('TUPLE1','TUPLE2','TUPLE3'):
                n=int(name[-1]);values=tuple(stack[-n:]);del stack[-n:];stack.append(values)
            elif name=='APPENDS':
                values=marked();stack[-1].extend(values)
            elif name=='APPEND':
                value=stack.pop();stack[-1].append(value)
            elif name=='SETITEMS':
                values=marked()
                if len(values)%2:raise ValueError('Odd dictionary items')
                stack[-1].update(zip(values[::2],values[1::2]))
            elif name=='SETITEM':
                value=stack.pop();key=stack.pop();stack[-1][key]=value
            elif name=='GLOBAL':stack.append(Opaque('global',tuple(arg.split(' '))))
            elif name=='STACK_GLOBAL':
                symbol=stack.pop();module=stack.pop();stack.append(Opaque('global',(module,symbol)))
            elif name=='REDUCE':
                args=stack.pop();function=stack.pop()
                kind='reduce'
                if isinstance(function,Opaque) and function.kind=='global':
                    if function.args==('numpy','dtype'):kind='dtype'
                    elif function.args in (('numpy.core.multiarray','_reconstruct'),('numpy._core.multiarray','_reconstruct')):kind='array'
                stack.append(Opaque(kind,args))
            elif name=='BUILD':
                state=stack.pop()
                if not isinstance(stack[-1],Opaque):raise ValueError('Unsupported pickle BUILD')
                stack[-1].state=state
            else:raise ValueError('Unsupported pickle opcode '+name)
    except (IndexError,KeyError,TypeError,AttributeError) as exc:
        raise ValueError('Malformed pickle structure') from exc
    if not stopped:raise ValueError('Missing pickle STOP')
    active=set()
    def resolve(value,depth=0):
        if depth>100:raise ValueError('Pickle nesting bound exceeded')
        if isinstance(value,Opaque):
            if value.kind!='array':return value
            state=value.state
            if not isinstance(state,tuple) or len(state)!=5:raise ValueError('Invalid numpy array state')
            version,shape,dtype,fortran,data=state
            if not isinstance(dtype,Opaque) or dtype.kind!='dtype' or not isinstance(dtype.args,tuple) or not dtype.args or dtype.args[0] not in ('f4','f8'):
                raise ValueError('Only numeric float dtype is accepted')
            if not isinstance(dtype.state,tuple) or len(dtype.state)!=8 or dtype.state[1] not in ('<','>','='):
                raise ValueError('Invalid dtype state')
            if (version!=1 or not isinstance(shape,tuple) or not 1<=len(shape)<=3 or
                any(type(n) is not int or n<1 for n in shape) or math.prod(shape)>4096 or
                type(fortran) is not bool or not isinstance(data,bytes)):
                raise ValueError('Invalid numeric array shape/payload')
            dt=np.dtype(dtype.state[1]+dtype.args[0])
            if len(data)!=math.prod(shape)*dt.itemsize:raise ValueError('Numeric payload length mismatch')
            return np.frombuffer(data,dtype=dt).reshape(shape,order='F' if fortran else 'C').copy()
        if isinstance(value,(dict,list,tuple)):
            if id(value) in active:raise ValueError('Cyclic pickle data')
            active.add(id(value))
            if isinstance(value,dict):result={resolve(k,depth+1):resolve(v,depth+1) for k,v in value.items()}
            elif isinstance(value,list):result=[resolve(v,depth+1) for v in value]
            else:result=tuple(resolve(v,depth+1) for v in value)
            active.remove(id(value));return result
        return value
    return resolve(result)
