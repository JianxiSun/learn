import tvm
import tvm.testing
from tvm import te
import numpy
import timeit

M = 1024
K = 1024
N = 1024

dtype = "float32"

target = "llvm"
dev = tvm.device(target, 0)

a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

np_repeat = 100
np_running_time = timeit.timeit(
    setup='import numpy\n'
          'M = ' + str(M) + '\n'
                            'K = ' + str(K) + '\n'
                                              'N = ' + str(N) + '\n'
                                                                'dtype = "float32"\n'
                                                                'a = numpy.random.rand(M, K).astype(dtype)\n'
                                                                'b = numpy.random.rand(K, N).astype(dtype)\n',
    stmt='answer = numpy.dot(a,b)',
    number=np_repeat
)

print("Numpy running time: %f" % (np_running_time / np_repeat))

answer = numpy.dot(a.asnumpy(), b.asnumpy())

# Baseline

k = te.reduce_axis((0, K), 'k')
A = te.placeholder((M, K), name='A')
B = te.placeholder((K, N), name='B')
C = te.compute(
    (M, N),
    lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
    name='C'
)

s = te.create_schedule(C.op)
func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=1)
print("Baseline: %f" % evaluator(a, b, c).mean)

# print(tvm.lower(s, [A, B, C], simple_mode=True))

# 分块

bn = 32
kfactor = 4
s = te.create_schedule(C.op)

mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(kaxis,) = s[C].op.reduce_axis
ko, ki = s[C].split(kaxis, factor=kfactor)

s[C].reorder(mo, no, ko, ki, mi, ni)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=10)
print("Opt1: %f" % evaluator(a, b, c).mean)

# print(tvm.lower(s, [A, B, C], simple_mode=True))

# 向量化

s = te.create_schedule(C.op)
mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(kaxis,) = s[C].op.reduce_axis
ko, ki = s[C].split(kaxis, factor=kfactor)

s[C].reorder(mo, no, ko, ki, mi, ni)

s[C].vectorize(ni)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=10)
print("Opt2: %f" % evaluator(a, b, c).mean)

# print(tvm.lower(s, [A, B, C], simple_mode=True))

# 循环置换

s = te.create_schedule(C.op)
mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(kaxis,) = s[C].op.reduce_axis
ko, ki = s[C].split(kaxis, factor=kfactor)

s[C].reorder(mo, no, ko, mi, ki, ni)
s[C].vectorize(ni)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=10)
print("Opt3: %f" % evaluator(a, b, c).mean)

# print(tvm.lower(s, [A, B, C], simple_mode=True))

# 数组打包

packedB = te.compute(
    (N / bn, K, bn), lambda bigN, k, littleN: B[k, bigN * bn + littleN], name="packedB"
)

C = te.compute(
    (M, N),
    lambda m, n: te.sum(A[m, k] * packedB[n // bn, k, tvm.tir.indexmod(n, bn)], axis=k),
    name="C"
)

s = te.create_schedule(C.op)

mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(kaxis,) = s[C].op.reduce_axis
ko, ki = s[C].split(kaxis, factor=kfactor)

s[C].reorder(mo, no, ko, mi, ki, ni)
s[C].vectorize(ni)

bigN, _, littleN = s[packedB].op.axis
s[packedB].vectorize(littleN)
s[packedB].parallel(bigN)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=10)
print("Opt4: %f" % evaluator(a, b, c).mean)

# print(tvm.lower(s, [A, B, C], simple_mode=True))

# 块的写缓存

s = te.create_schedule(C.op)

CC = s.cache_write(C, "global")
mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

s[CC].compute_at(s[C], no)

mc, nc = s[CC].op.axis

(kaxis,) = s[CC].op.reduce_axis
ko, ki = s[CC].split(kaxis, factor=kfactor)
s[CC].reorder(ko, mc, ki, nc)
s[CC].vectorize(nc)

s[CC].unroll(ki)

bigN, _, littleN = s[packedB].op.axis
s[packedB].vectorize(littleN)
s[packedB].parallel(bigN)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=10)
print("Opt5: %f" % evaluator(a, b, c).mean)

# print(tvm.lower(s, [A, B, C], simple_mode=True))

# 并行化

s = te.create_schedule(C.op)

CC = s.cache_write(C, "global")
mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

s[CC].compute_at(s[C], no)

mc, nc = s[CC].op.axis

(kaxis,) = s[CC].op.reduce_axis
ko, ki = s[CC].split(kaxis, factor=kfactor)
s[CC].reorder(ko, mc, ki, nc)
s[CC].vectorize(nc)
s[CC].unroll(ki)

s[C].parallel(mo)

bigN, _, littleN = s[packedB].op.axis
s[packedB].vectorize(littleN)
s[packedB].parallel(bigN)

func = tvm.build(s, [A, B, C], target=target, name="mmult")
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=50)
print("Opt6: %f" % evaluator(a, b, c).mean)

# print(tvm.lower(s, [A, B, C], simple_mode=True))
