հ
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??

u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	P?*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	P?*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
??*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:?*
dtype0
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
:
*
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
:
*
dtype0
~
conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

* 
shared_nameconv1d_3/kernel
w
#conv1d_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_3/kernel*"
_output_shapes
:

*
dtype0
r
conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv1d_3/bias
k
!conv1d_3/bias/Read/ReadVariableOpReadVariableOpconv1d_3/bias*
_output_shapes
:
*
dtype0
~
conv1d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_nameconv1d_4/kernel
w
#conv1d_4/kernel/Read/ReadVariableOpReadVariableOpconv1d_4/kernel*"
_output_shapes
:
*
dtype0
r
conv1d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_4/bias
k
!conv1d_4/bias/Read/ReadVariableOpReadVariableOpconv1d_4/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	P?*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	P?*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_1/kernel/m
?
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
??*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_2/kernel/m
?
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m* 
_output_shapes
:
??*
dtype0

Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_2/bias/m
x
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv1d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/conv1d_2/kernel/m
?
*Adam/conv1d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/m*"
_output_shapes
:
*
dtype0
?
Adam/conv1d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv1d_2/bias/m
y
(Adam/conv1d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/m*
_output_shapes
:
*
dtype0
?
Adam/conv1d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*'
shared_nameAdam/conv1d_3/kernel/m
?
*Adam/conv1d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/kernel/m*"
_output_shapes
:

*
dtype0
?
Adam/conv1d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv1d_3/bias/m
y
(Adam/conv1d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/bias/m*
_output_shapes
:
*
dtype0
?
Adam/conv1d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/conv1d_4/kernel/m
?
*Adam/conv1d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_4/kernel/m*"
_output_shapes
:
*
dtype0
?
Adam/conv1d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_4/bias/m
y
(Adam/conv1d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_4/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	P?*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	P?*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_1/kernel/v
?
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
??*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_2/kernel/v
?
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v* 
_output_shapes
:
??*
dtype0

Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_2/bias/v
x
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv1d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/conv1d_2/kernel/v
?
*Adam/conv1d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/v*"
_output_shapes
:
*
dtype0
?
Adam/conv1d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv1d_2/bias/v
y
(Adam/conv1d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/v*
_output_shapes
:
*
dtype0
?
Adam/conv1d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*'
shared_nameAdam/conv1d_3/kernel/v
?
*Adam/conv1d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/kernel/v*"
_output_shapes
:

*
dtype0
?
Adam/conv1d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv1d_3/bias/v
y
(Adam/conv1d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_3/bias/v*
_output_shapes
:
*
dtype0
?
Adam/conv1d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/conv1d_4/kernel/v
?
*Adam/conv1d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_4/kernel/v*"
_output_shapes
:
*
dtype0
?
Adam/conv1d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv1d_4/bias/v
y
(Adam/conv1d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_4/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?J
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?J
value?JB?J B?I
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
	optimizer

signatures
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
%
#_self_saveable_object_factories
?

kernel
bias
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
w
#_self_saveable_object_factories
trainable_variables
 	variables
!regularization_losses
"	keras_api
?

#kernel
$bias
#%_self_saveable_object_factories
&trainable_variables
'	variables
(regularization_losses
)	keras_api
w
#*_self_saveable_object_factories
+trainable_variables
,	variables
-regularization_losses
.	keras_api
?

/kernel
0bias
#1_self_saveable_object_factories
2trainable_variables
3	variables
4regularization_losses
5	keras_api
4
#6_self_saveable_object_factories
7	keras_api
?

8kernel
9bias
#:_self_saveable_object_factories
;trainable_variables
<	variables
=regularization_losses
>	keras_api
4
#?_self_saveable_object_factories
@	keras_api
4
#A_self_saveable_object_factories
B	keras_api
4
#C_self_saveable_object_factories
D	keras_api
?

Ekernel
Fbias
#G_self_saveable_object_factories
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
?

Lkernel
Mbias
#N_self_saveable_object_factories
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
4
#S_self_saveable_object_factories
T	keras_api
?
Uiter

Vbeta_1

Wbeta_2
	Xdecay
Ylearning_ratem?m?#m?$m?/m?0m?8m?9m?Em?Fm?Lm?Mm?v?v?#v?$v?/v?0v?8v?9v?Ev?Fv?Lv?Mv?
 
 
V
0
1
#2
$3
/4
05
86
97
E8
F9
L10
M11
V
0
1
#2
$3
/4
05
86
97
E8
F9
L10
M11
 
?
Zmetrics
trainable_variables
	variables
[layer_metrics
\non_trainable_variables
]layer_regularization_losses
regularization_losses

^layers
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
?
_metrics
trainable_variables
	variables
`layer_metrics
anon_trainable_variables
blayer_regularization_losses
regularization_losses

clayers
 
 
 
 
?
dmetrics
trainable_variables
 	variables
elayer_metrics
fnon_trainable_variables
glayer_regularization_losses
!regularization_losses

hlayers
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
 
?
imetrics
&trainable_variables
'	variables
jlayer_metrics
knon_trainable_variables
llayer_regularization_losses
(regularization_losses

mlayers
 
 
 
 
?
nmetrics
+trainable_variables
,	variables
olayer_metrics
pnon_trainable_variables
qlayer_regularization_losses
-regularization_losses

rlayers
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

/0
01
 
?
smetrics
2trainable_variables
3	variables
tlayer_metrics
unon_trainable_variables
vlayer_regularization_losses
4regularization_losses

wlayers
 
 
[Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

80
91

80
91
 
?
xmetrics
;trainable_variables
<	variables
ylayer_metrics
znon_trainable_variables
{layer_regularization_losses
=regularization_losses

|layers
 
 
 
 
 
 
[Y
VARIABLE_VALUEconv1d_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

E0
F1

E0
F1
 
?
}metrics
Htrainable_variables
I	variables
~layer_metrics
non_trainable_variables
 ?layer_regularization_losses
Jregularization_losses
?layers
[Y
VARIABLE_VALUEconv1d_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1

L0
M1
 
?
?metrics
Otrainable_variables
P	variables
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
Qregularization_losses
?layers
 
 
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 
 
 
f
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_3/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_3/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_4/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_4/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_3/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_3/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_4/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_4/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_3Placeholder*'
_output_shapes
:?????????P*
dtype0*
shape:?????????P
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3dense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biasconv1d_4/kernelconv1d_4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *-
f(R&
$__inference_signature_wrapper_201058
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp#conv1d_3/kernel/Read/ReadVariableOp!conv1d_3/bias/Read/ReadVariableOp#conv1d_4/kernel/Read/ReadVariableOp!conv1d_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp*Adam/conv1d_2/kernel/m/Read/ReadVariableOp(Adam/conv1d_2/bias/m/Read/ReadVariableOp*Adam/conv1d_3/kernel/m/Read/ReadVariableOp(Adam/conv1d_3/bias/m/Read/ReadVariableOp*Adam/conv1d_4/kernel/m/Read/ReadVariableOp(Adam/conv1d_4/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp*Adam/conv1d_2/kernel/v/Read/ReadVariableOp(Adam/conv1d_2/bias/v/Read/ReadVariableOp*Adam/conv1d_3/kernel/v/Read/ReadVariableOp(Adam/conv1d_3/bias/v/Read/ReadVariableOp*Adam/conv1d_4/kernel/v/Read/ReadVariableOp(Adam/conv1d_4/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *(
f#R!
__inference__traced_save_201615
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biasconv1d_4/kernelconv1d_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/conv1d_2/kernel/mAdam/conv1d_2/bias/mAdam/conv1d_3/kernel/mAdam/conv1d_3/bias/mAdam/conv1d_4/kernel/mAdam/conv1d_4/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/conv1d_2/kernel/vAdam/conv1d_2/bias/vAdam/conv1d_3/kernel/vAdam/conv1d_3/bias/vAdam/conv1d_4/kernel/vAdam/conv1d_4/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *+
f&R$
"__inference__traced_restore_201760׳	
?
?
D__inference_conv1d_3_layer_call_and_return_conditional_losses_201424

inputsA
+conv1d_expanddims_1_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:

*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:

2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????
*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
2	
BiasAddp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????
2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
?
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_201297

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_201309

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_2_layer_call_fn_201319

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2007772
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_201344

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_conv1d_3_layer_call_and_return_conditional_losses_200627

inputsA
+conv1d_expanddims_1_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:

*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:

2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????
*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
2	
BiasAddp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????
2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
(__inference_model_2_layer_call_fn_201272

inputs
unknown:	P?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:

	unknown_6:

	unknown_7:


	unknown_8:

	unknown_9:


unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_2008692
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
)__inference_conv1d_3_layer_call_fn_201433

inputs
unknown:


	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_2006272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
??
?
"__inference__traced_restore_201760
file_prefix0
assignvariableop_dense_kernel:	P?,
assignvariableop_1_dense_bias:	?5
!assignvariableop_2_dense_1_kernel:
??.
assignvariableop_3_dense_1_bias:	?5
!assignvariableop_4_dense_2_kernel:
??.
assignvariableop_5_dense_2_bias:	?8
"assignvariableop_6_conv1d_2_kernel:
.
 assignvariableop_7_conv1d_2_bias:
8
"assignvariableop_8_conv1d_3_kernel:

.
 assignvariableop_9_conv1d_3_bias:
9
#assignvariableop_10_conv1d_4_kernel:
/
!assignvariableop_11_conv1d_4_bias:'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: :
'assignvariableop_21_adam_dense_kernel_m:	P?4
%assignvariableop_22_adam_dense_bias_m:	?=
)assignvariableop_23_adam_dense_1_kernel_m:
??6
'assignvariableop_24_adam_dense_1_bias_m:	?=
)assignvariableop_25_adam_dense_2_kernel_m:
??6
'assignvariableop_26_adam_dense_2_bias_m:	?@
*assignvariableop_27_adam_conv1d_2_kernel_m:
6
(assignvariableop_28_adam_conv1d_2_bias_m:
@
*assignvariableop_29_adam_conv1d_3_kernel_m:

6
(assignvariableop_30_adam_conv1d_3_bias_m:
@
*assignvariableop_31_adam_conv1d_4_kernel_m:
6
(assignvariableop_32_adam_conv1d_4_bias_m::
'assignvariableop_33_adam_dense_kernel_v:	P?4
%assignvariableop_34_adam_dense_bias_v:	?=
)assignvariableop_35_adam_dense_1_kernel_v:
??6
'assignvariableop_36_adam_dense_1_bias_v:	?=
)assignvariableop_37_adam_dense_2_kernel_v:
??6
'assignvariableop_38_adam_dense_2_bias_v:	?@
*assignvariableop_39_adam_conv1d_2_kernel_v:
6
(assignvariableop_40_adam_conv1d_2_bias_v:
@
*assignvariableop_41_adam_conv1d_3_kernel_v:

6
(assignvariableop_42_adam_conv1d_3_bias_v:
@
*assignvariableop_43_adam_conv1d_4_kernel_v:
6
(assignvariableop_44_adam_conv1d_4_bias_v:
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv1d_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv1d_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv1d_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv1d_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv1d_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv1d_4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp%assignvariableop_22_adam_dense_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_2_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_2_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv1d_2_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv1d_2_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv1d_3_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv1d_3_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv1d_4_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv1d_4_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_dense_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp%assignvariableop_34_adam_dense_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_1_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_1_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_2_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_2_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv1d_2_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv1d_2_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv1d_3_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv1d_3_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv1d_4_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv1d_4_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45f
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_46?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
F
*__inference_dropout_3_layer_call_fn_201361

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2005632
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_conv1d_4_layer_call_and_return_conditional_losses_201448

inputsA
+conv1d_expanddims_1_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAddp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
?7
?
C__inference_model_2_layer_call_and_return_conditional_losses_200657

inputs
dense_200529:	P?
dense_200531:	?"
dense_1_200553:
??
dense_1_200555:	?"
dense_2_200576:
??
dense_2_200578:	?%
conv1d_2_200599:

conv1d_2_200601:
%
conv1d_3_200628:


conv1d_3_200630:
%
conv1d_4_200649:

conv1d_4_200651:
identity?? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall? conv1d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_200529dense_200531*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2005282
dense/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2005392
dropout_2/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_1_200553dense_1_200555*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2005522!
dense_1/StatefulPartitionedCall?
dropout_3/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2005632
dropout_3/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_2_200576dense_2_200578*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2005752!
dense_2/StatefulPartitionedCall?
tf.reshape_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????     2
tf.reshape_2/Reshape/shape?
tf.reshape_2/ReshapeReshape(dense_2/StatefulPartitionedCall:output:0#tf.reshape_2/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
tf.reshape_2/Reshape?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCalltf.reshape_2/Reshape:output:0conv1d_2_200599conv1d_2_200601*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_2005982"
 conv1d_2/StatefulPartitionedCally
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_2/Mul/y?
tf.math.multiply_2/MulMul)conv1d_2/StatefulPartitionedCall:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*,
_output_shapes
:??????????
2
tf.math.multiply_2/Mulo
tf.math.add_2/Add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
tf.math.add_2/Add/y?
tf.math.add_2/AddAddV2tf.math.multiply_2/Mul:z:0tf.math.add_2/Add/y:output:0*
T0*,
_output_shapes
:??????????
2
tf.math.add_2/Add?
*tf.clip_by_value_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*tf.clip_by_value_2/clip_by_value/Minimum/y?
(tf.clip_by_value_2/clip_by_value/MinimumMinimumtf.math.add_2/Add:z:03tf.clip_by_value_2/clip_by_value/Minimum/y:output:0*
T0*,
_output_shapes
:??????????
2*
(tf.clip_by_value_2/clip_by_value/Minimum?
"tf.clip_by_value_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_2/clip_by_value/y?
 tf.clip_by_value_2/clip_by_valueMaximum,tf.clip_by_value_2/clip_by_value/Minimum:z:0+tf.clip_by_value_2/clip_by_value/y:output:0*
T0*,
_output_shapes
:??????????
2"
 tf.clip_by_value_2/clip_by_value?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_2/clip_by_value:z:0conv1d_3_200628conv1d_3_200630*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_2006272"
 conv1d_3/StatefulPartitionedCall?
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0conv1d_4_200649conv1d_4_200651*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_2006482"
 conv1d_4/StatefulPartitionedCall?
tf.reshape_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2
tf.reshape_3/Reshape/shape?
tf.reshape_3/ReshapeReshape)conv1d_4/StatefulPartitionedCall:output:0#tf.reshape_3/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2
tf.reshape_3/Reshapey
IdentityIdentitytf.reshape_3/Reshape:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : : : 2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
D__inference_conv1d_2_layer_call_and_return_conditional_losses_200598

inputsA
+conv1d_expanddims_1_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????
*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
2	
BiasAddp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????
2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_200744

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_201356

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_3_layer_call_fn_201366

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2007442
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_conv1d_2_layer_call_fn_201409

inputs
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_2005982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_200777

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_2_layer_call_fn_201314

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2005392
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_1_layer_call_fn_201339

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2005522
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_2_layer_call_fn_201385

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2005752
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_conv1d_2_layer_call_and_return_conditional_losses_201400

inputsA
+conv1d_expanddims_1_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????
*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
2	
BiasAddp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????
2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
A__inference_dense_layer_call_and_return_conditional_losses_201283

inputs1
matmul_readvariableop_resource:	P?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	P?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?]
?
__inference__traced_save_201615
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop,
(savev2_conv1d_3_bias_read_readvariableop.
*savev2_conv1d_4_kernel_read_readvariableop,
(savev2_conv1d_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop5
1savev2_adam_conv1d_2_kernel_m_read_readvariableop3
/savev2_adam_conv1d_2_bias_m_read_readvariableop5
1savev2_adam_conv1d_3_kernel_m_read_readvariableop3
/savev2_adam_conv1d_3_bias_m_read_readvariableop5
1savev2_adam_conv1d_4_kernel_m_read_readvariableop3
/savev2_adam_conv1d_4_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop5
1savev2_adam_conv1d_2_kernel_v_read_readvariableop3
/savev2_adam_conv1d_2_bias_v_read_readvariableop5
1savev2_adam_conv1d_3_kernel_v_read_readvariableop3
/savev2_adam_conv1d_3_bias_v_read_readvariableop5
1savev2_adam_conv1d_4_kernel_v_read_readvariableop3
/savev2_adam_conv1d_4_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop*savev2_conv1d_4_kernel_read_readvariableop(savev2_conv1d_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop1savev2_adam_conv1d_2_kernel_m_read_readvariableop/savev2_adam_conv1d_2_bias_m_read_readvariableop1savev2_adam_conv1d_3_kernel_m_read_readvariableop/savev2_adam_conv1d_3_bias_m_read_readvariableop1savev2_adam_conv1d_4_kernel_m_read_readvariableop/savev2_adam_conv1d_4_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop1savev2_adam_conv1d_2_kernel_v_read_readvariableop/savev2_adam_conv1d_2_bias_v_read_readvariableop1savev2_adam_conv1d_3_kernel_v_read_readvariableop/savev2_adam_conv1d_3_bias_v_read_readvariableop1savev2_adam_conv1d_4_kernel_v_read_readvariableop/savev2_adam_conv1d_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	P?:?:
??:?:
??:?:
:
:

:
:
:: : : : : : : : : :	P?:?:
??:?:
??:?:
:
:

:
:
::	P?:?:
??:?:
??:?:
:
:

:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	P?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:($
"
_output_shapes
:
: 

_output_shapes
:
:(	$
"
_output_shapes
:

: 


_output_shapes
:
:($
"
_output_shapes
:
: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	P?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:($
"
_output_shapes
:
: 

_output_shapes
:
:($
"
_output_shapes
:

: 

_output_shapes
:
:( $
"
_output_shapes
:
: !

_output_shapes
::%"!

_output_shapes
:	P?:!#

_output_shapes	
:?:&$"
 
_output_shapes
:
??:!%

_output_shapes	
:?:&&"
 
_output_shapes
:
??:!'

_output_shapes	
:?:(($
"
_output_shapes
:
: )

_output_shapes
:
:(*$
"
_output_shapes
:

: +

_output_shapes
:
:(,$
"
_output_shapes
:
: -

_output_shapes
::.

_output_shapes
: 
?:
?
C__inference_model_2_layer_call_and_return_conditional_losses_201021
input_3
dense_200976:	P?
dense_200978:	?"
dense_1_200982:
??
dense_1_200984:	?"
dense_2_200988:
??
dense_2_200990:	?%
conv1d_2_200995:

conv1d_2_200997:
%
conv1d_3_201008:


conv1d_3_201010:
%
conv1d_4_201013:

conv1d_4_201015:
identity?? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall? conv1d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_3dense_200976dense_200978*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2005282
dense/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2007772#
!dropout_2/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_1_200982dense_1_200984*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2005522!
dense_1/StatefulPartitionedCall?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2007442#
!dropout_3/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_2_200988dense_2_200990*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2005752!
dense_2/StatefulPartitionedCall?
tf.reshape_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????     2
tf.reshape_2/Reshape/shape?
tf.reshape_2/ReshapeReshape(dense_2/StatefulPartitionedCall:output:0#tf.reshape_2/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
tf.reshape_2/Reshape?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCalltf.reshape_2/Reshape:output:0conv1d_2_200995conv1d_2_200997*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_2005982"
 conv1d_2/StatefulPartitionedCally
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_2/Mul/y?
tf.math.multiply_2/MulMul)conv1d_2/StatefulPartitionedCall:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*,
_output_shapes
:??????????
2
tf.math.multiply_2/Mulo
tf.math.add_2/Add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
tf.math.add_2/Add/y?
tf.math.add_2/AddAddV2tf.math.multiply_2/Mul:z:0tf.math.add_2/Add/y:output:0*
T0*,
_output_shapes
:??????????
2
tf.math.add_2/Add?
*tf.clip_by_value_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*tf.clip_by_value_2/clip_by_value/Minimum/y?
(tf.clip_by_value_2/clip_by_value/MinimumMinimumtf.math.add_2/Add:z:03tf.clip_by_value_2/clip_by_value/Minimum/y:output:0*
T0*,
_output_shapes
:??????????
2*
(tf.clip_by_value_2/clip_by_value/Minimum?
"tf.clip_by_value_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_2/clip_by_value/y?
 tf.clip_by_value_2/clip_by_valueMaximum,tf.clip_by_value_2/clip_by_value/Minimum:z:0+tf.clip_by_value_2/clip_by_value/y:output:0*
T0*,
_output_shapes
:??????????
2"
 tf.clip_by_value_2/clip_by_value?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_2/clip_by_value:z:0conv1d_3_201008conv1d_3_201010*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_2006272"
 conv1d_3/StatefulPartitionedCall?
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0conv1d_4_201013conv1d_4_201015*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_2006482"
 conv1d_4/StatefulPartitionedCall?
tf.reshape_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2
tf.reshape_3/Reshape/shape?
tf.reshape_3/ReshapeReshape)conv1d_4/StatefulPartitionedCall:output:0#tf.reshape_3/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2
tf.reshape_3/Reshapey
IdentityIdentitytf.reshape_3/Reshape:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : : : 2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:P L
'
_output_shapes
:?????????P
!
_user_specified_name	input_3
?
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_200539

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_conv1d_4_layer_call_fn_201457

inputs
unknown:

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_2006482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
A__inference_dense_layer_call_and_return_conditional_losses_200528

inputs1
matmul_readvariableop_resource:	P?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	P?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?:
?
C__inference_model_2_layer_call_and_return_conditional_losses_200869

inputs
dense_200824:	P?
dense_200826:	?"
dense_1_200830:
??
dense_1_200832:	?"
dense_2_200836:
??
dense_2_200838:	?%
conv1d_2_200843:

conv1d_2_200845:
%
conv1d_3_200856:


conv1d_3_200858:
%
conv1d_4_200861:

conv1d_4_200863:
identity?? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall? conv1d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_200824dense_200826*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2005282
dense/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2007772#
!dropout_2/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_1_200830dense_1_200832*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2005522!
dense_1/StatefulPartitionedCall?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2007442#
!dropout_3/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_2_200836dense_2_200838*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2005752!
dense_2/StatefulPartitionedCall?
tf.reshape_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????     2
tf.reshape_2/Reshape/shape?
tf.reshape_2/ReshapeReshape(dense_2/StatefulPartitionedCall:output:0#tf.reshape_2/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
tf.reshape_2/Reshape?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCalltf.reshape_2/Reshape:output:0conv1d_2_200843conv1d_2_200845*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_2005982"
 conv1d_2/StatefulPartitionedCally
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_2/Mul/y?
tf.math.multiply_2/MulMul)conv1d_2/StatefulPartitionedCall:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*,
_output_shapes
:??????????
2
tf.math.multiply_2/Mulo
tf.math.add_2/Add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
tf.math.add_2/Add/y?
tf.math.add_2/AddAddV2tf.math.multiply_2/Mul:z:0tf.math.add_2/Add/y:output:0*
T0*,
_output_shapes
:??????????
2
tf.math.add_2/Add?
*tf.clip_by_value_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*tf.clip_by_value_2/clip_by_value/Minimum/y?
(tf.clip_by_value_2/clip_by_value/MinimumMinimumtf.math.add_2/Add:z:03tf.clip_by_value_2/clip_by_value/Minimum/y:output:0*
T0*,
_output_shapes
:??????????
2*
(tf.clip_by_value_2/clip_by_value/Minimum?
"tf.clip_by_value_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_2/clip_by_value/y?
 tf.clip_by_value_2/clip_by_valueMaximum,tf.clip_by_value_2/clip_by_value/Minimum:z:0+tf.clip_by_value_2/clip_by_value/y:output:0*
T0*,
_output_shapes
:??????????
2"
 tf.clip_by_value_2/clip_by_value?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_2/clip_by_value:z:0conv1d_3_200856conv1d_3_200858*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_2006272"
 conv1d_3/StatefulPartitionedCall?
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0conv1d_4_200861conv1d_4_200863*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_2006482"
 conv1d_4/StatefulPartitionedCall?
tf.reshape_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2
tf.reshape_3/Reshape/shape?
tf.reshape_3/ReshapeReshape)conv1d_4/StatefulPartitionedCall:output:0#tf.reshape_3/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2
tf.reshape_3/Reshapey
IdentityIdentitytf.reshape_3/Reshape:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : : : 2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
(__inference_model_2_layer_call_fn_200684
input_3
unknown:	P?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:

	unknown_6:

	unknown_7:


	unknown_8:

	unknown_9:


unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_2006572
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????P
!
_user_specified_name	input_3
?
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_200563

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_conv1d_4_layer_call_and_return_conditional_losses_200648

inputsA
+conv1d_expanddims_1_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAddp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????

 
_user_specified_nameinputs
?o
?
!__inference__wrapped_model_200510
input_3?
,model_2_dense_matmul_readvariableop_resource:	P?<
-model_2_dense_biasadd_readvariableop_resource:	?B
.model_2_dense_1_matmul_readvariableop_resource:
??>
/model_2_dense_1_biasadd_readvariableop_resource:	?B
.model_2_dense_2_matmul_readvariableop_resource:
??>
/model_2_dense_2_biasadd_readvariableop_resource:	?R
<model_2_conv1d_2_conv1d_expanddims_1_readvariableop_resource:
>
0model_2_conv1d_2_biasadd_readvariableop_resource:
R
<model_2_conv1d_3_conv1d_expanddims_1_readvariableop_resource:

>
0model_2_conv1d_3_biasadd_readvariableop_resource:
R
<model_2_conv1d_4_conv1d_expanddims_1_readvariableop_resource:
>
0model_2_conv1d_4_biasadd_readvariableop_resource:
identity??'model_2/conv1d_2/BiasAdd/ReadVariableOp?3model_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?'model_2/conv1d_3/BiasAdd/ReadVariableOp?3model_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?'model_2/conv1d_4/BiasAdd/ReadVariableOp?3model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp?$model_2/dense/BiasAdd/ReadVariableOp?#model_2/dense/MatMul/ReadVariableOp?&model_2/dense_1/BiasAdd/ReadVariableOp?%model_2/dense_1/MatMul/ReadVariableOp?&model_2/dense_2/BiasAdd/ReadVariableOp?%model_2/dense_2/MatMul/ReadVariableOp?
#model_2/dense/MatMul/ReadVariableOpReadVariableOp,model_2_dense_matmul_readvariableop_resource*
_output_shapes
:	P?*
dtype02%
#model_2/dense/MatMul/ReadVariableOp?
model_2/dense/MatMulMatMulinput_3+model_2/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_2/dense/MatMul?
$model_2/dense/BiasAdd/ReadVariableOpReadVariableOp-model_2_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$model_2/dense/BiasAdd/ReadVariableOp?
model_2/dense/BiasAddBiasAddmodel_2/dense/MatMul:product:0,model_2/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_2/dense/BiasAdd?
model_2/dense/ReluRelumodel_2/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_2/dense/Relu?
model_2/dropout_2/IdentityIdentity model_2/dense/Relu:activations:0*
T0*(
_output_shapes
:??????????2
model_2/dropout_2/Identity?
%model_2/dense_1/MatMul/ReadVariableOpReadVariableOp.model_2_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%model_2/dense_1/MatMul/ReadVariableOp?
model_2/dense_1/MatMulMatMul#model_2/dropout_2/Identity:output:0-model_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_2/dense_1/MatMul?
&model_2/dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&model_2/dense_1/BiasAdd/ReadVariableOp?
model_2/dense_1/BiasAddBiasAdd model_2/dense_1/MatMul:product:0.model_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_2/dense_1/BiasAdd?
model_2/dense_1/SigmoidSigmoid model_2/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_2/dense_1/Sigmoid?
model_2/dropout_3/IdentityIdentitymodel_2/dense_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
model_2/dropout_3/Identity?
%model_2/dense_2/MatMul/ReadVariableOpReadVariableOp.model_2_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%model_2/dense_2/MatMul/ReadVariableOp?
model_2/dense_2/MatMulMatMul#model_2/dropout_3/Identity:output:0-model_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_2/dense_2/MatMul?
&model_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&model_2/dense_2/BiasAdd/ReadVariableOp?
model_2/dense_2/BiasAddBiasAdd model_2/dense_2/MatMul:product:0.model_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_2/dense_2/BiasAdd?
"model_2/tf.reshape_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????     2$
"model_2/tf.reshape_2/Reshape/shape?
model_2/tf.reshape_2/ReshapeReshape model_2/dense_2/BiasAdd:output:0+model_2/tf.reshape_2/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
model_2/tf.reshape_2/Reshape?
&model_2/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&model_2/conv1d_2/conv1d/ExpandDims/dim?
"model_2/conv1d_2/conv1d/ExpandDims
ExpandDims%model_2/tf.reshape_2/Reshape:output:0/model_2/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2$
"model_2/conv1d_2/conv1d/ExpandDims?
3model_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<model_2_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype025
3model_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
(model_2/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_2/conv1d_2/conv1d/ExpandDims_1/dim?
$model_2/conv1d_2/conv1d/ExpandDims_1
ExpandDims;model_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:01model_2/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2&
$model_2/conv1d_2/conv1d/ExpandDims_1?
model_2/conv1d_2/conv1dConv2D+model_2/conv1d_2/conv1d/ExpandDims:output:0-model_2/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
*
paddingSAME*
strides
2
model_2/conv1d_2/conv1d?
model_2/conv1d_2/conv1d/SqueezeSqueeze model_2/conv1d_2/conv1d:output:0*
T0*,
_output_shapes
:??????????
*
squeeze_dims

?????????2!
model_2/conv1d_2/conv1d/Squeeze?
'model_2/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'model_2/conv1d_2/BiasAdd/ReadVariableOp?
model_2/conv1d_2/BiasAddBiasAdd(model_2/conv1d_2/conv1d/Squeeze:output:0/model_2/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
2
model_2/conv1d_2/BiasAdd?
 model_2/tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 model_2/tf.math.multiply_2/Mul/y?
model_2/tf.math.multiply_2/MulMul!model_2/conv1d_2/BiasAdd:output:0)model_2/tf.math.multiply_2/Mul/y:output:0*
T0*,
_output_shapes
:??????????
2 
model_2/tf.math.multiply_2/Mul
model_2/tf.math.add_2/Add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model_2/tf.math.add_2/Add/y?
model_2/tf.math.add_2/AddAddV2"model_2/tf.math.multiply_2/Mul:z:0$model_2/tf.math.add_2/Add/y:output:0*
T0*,
_output_shapes
:??????????
2
model_2/tf.math.add_2/Add?
2model_2/tf.clip_by_value_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2model_2/tf.clip_by_value_2/clip_by_value/Minimum/y?
0model_2/tf.clip_by_value_2/clip_by_value/MinimumMinimummodel_2/tf.math.add_2/Add:z:0;model_2/tf.clip_by_value_2/clip_by_value/Minimum/y:output:0*
T0*,
_output_shapes
:??????????
22
0model_2/tf.clip_by_value_2/clip_by_value/Minimum?
*model_2/tf.clip_by_value_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_2/tf.clip_by_value_2/clip_by_value/y?
(model_2/tf.clip_by_value_2/clip_by_valueMaximum4model_2/tf.clip_by_value_2/clip_by_value/Minimum:z:03model_2/tf.clip_by_value_2/clip_by_value/y:output:0*
T0*,
_output_shapes
:??????????
2*
(model_2/tf.clip_by_value_2/clip_by_value?
&model_2/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&model_2/conv1d_3/conv1d/ExpandDims/dim?
"model_2/conv1d_3/conv1d/ExpandDims
ExpandDims,model_2/tf.clip_by_value_2/clip_by_value:z:0/model_2/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
2$
"model_2/conv1d_3/conv1d/ExpandDims?
3model_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<model_2_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:

*
dtype025
3model_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?
(model_2/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_2/conv1d_3/conv1d/ExpandDims_1/dim?
$model_2/conv1d_3/conv1d/ExpandDims_1
ExpandDims;model_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:01model_2/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:

2&
$model_2/conv1d_3/conv1d/ExpandDims_1?
model_2/conv1d_3/conv1dConv2D+model_2/conv1d_3/conv1d/ExpandDims:output:0-model_2/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
*
paddingSAME*
strides
2
model_2/conv1d_3/conv1d?
model_2/conv1d_3/conv1d/SqueezeSqueeze model_2/conv1d_3/conv1d:output:0*
T0*,
_output_shapes
:??????????
*
squeeze_dims

?????????2!
model_2/conv1d_3/conv1d/Squeeze?
'model_2/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'model_2/conv1d_3/BiasAdd/ReadVariableOp?
model_2/conv1d_3/BiasAddBiasAdd(model_2/conv1d_3/conv1d/Squeeze:output:0/model_2/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
2
model_2/conv1d_3/BiasAdd?
&model_2/conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&model_2/conv1d_4/conv1d/ExpandDims/dim?
"model_2/conv1d_4/conv1d/ExpandDims
ExpandDims!model_2/conv1d_3/BiasAdd:output:0/model_2/conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
2$
"model_2/conv1d_4/conv1d/ExpandDims?
3model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<model_2_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype025
3model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp?
(model_2/conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_2/conv1d_4/conv1d/ExpandDims_1/dim?
$model_2/conv1d_4/conv1d/ExpandDims_1
ExpandDims;model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:01model_2/conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2&
$model_2/conv1d_4/conv1d/ExpandDims_1?
model_2/conv1d_4/conv1dConv2D+model_2/conv1d_4/conv1d/ExpandDims:output:0-model_2/conv1d_4/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_2/conv1d_4/conv1d?
model_2/conv1d_4/conv1d/SqueezeSqueeze model_2/conv1d_4/conv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2!
model_2/conv1d_4/conv1d/Squeeze?
'model_2/conv1d_4/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_2/conv1d_4/BiasAdd/ReadVariableOp?
model_2/conv1d_4/BiasAddBiasAdd(model_2/conv1d_4/conv1d/Squeeze:output:0/model_2/conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
model_2/conv1d_4/BiasAdd?
"model_2/tf.reshape_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2$
"model_2/tf.reshape_3/Reshape/shape?
model_2/tf.reshape_3/ReshapeReshape!model_2/conv1d_4/BiasAdd:output:0+model_2/tf.reshape_3/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2
model_2/tf.reshape_3/Reshape?
IdentityIdentity%model_2/tf.reshape_3/Reshape:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp(^model_2/conv1d_2/BiasAdd/ReadVariableOp4^model_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp(^model_2/conv1d_3/BiasAdd/ReadVariableOp4^model_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp(^model_2/conv1d_4/BiasAdd/ReadVariableOp4^model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp%^model_2/dense/BiasAdd/ReadVariableOp$^model_2/dense/MatMul/ReadVariableOp'^model_2/dense_1/BiasAdd/ReadVariableOp&^model_2/dense_1/MatMul/ReadVariableOp'^model_2/dense_2/BiasAdd/ReadVariableOp&^model_2/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : : : 2R
'model_2/conv1d_2/BiasAdd/ReadVariableOp'model_2/conv1d_2/BiasAdd/ReadVariableOp2j
3model_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp3model_2/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2R
'model_2/conv1d_3/BiasAdd/ReadVariableOp'model_2/conv1d_3/BiasAdd/ReadVariableOp2j
3model_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp3model_2/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp2R
'model_2/conv1d_4/BiasAdd/ReadVariableOp'model_2/conv1d_4/BiasAdd/ReadVariableOp2j
3model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp3model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp2L
$model_2/dense/BiasAdd/ReadVariableOp$model_2/dense/BiasAdd/ReadVariableOp2J
#model_2/dense/MatMul/ReadVariableOp#model_2/dense/MatMul/ReadVariableOp2P
&model_2/dense_1/BiasAdd/ReadVariableOp&model_2/dense_1/BiasAdd/ReadVariableOp2N
%model_2/dense_1/MatMul/ReadVariableOp%model_2/dense_1/MatMul/ReadVariableOp2P
&model_2/dense_2/BiasAdd/ReadVariableOp&model_2/dense_2/BiasAdd/ReadVariableOp2N
%model_2/dense_2/MatMul/ReadVariableOp%model_2/dense_2/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????P
!
_user_specified_name	input_3
?
?
(__inference_model_2_layer_call_fn_201243

inputs
unknown:	P?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:

	unknown_6:

	unknown_7:


	unknown_8:

	unknown_9:


unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_2006572
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?

?
C__inference_dense_2_layer_call_and_return_conditional_losses_200575

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?7
?
C__inference_model_2_layer_call_and_return_conditional_losses_200973
input_3
dense_200928:	P?
dense_200930:	?"
dense_1_200934:
??
dense_1_200936:	?"
dense_2_200940:
??
dense_2_200942:	?%
conv1d_2_200947:

conv1d_2_200949:
%
conv1d_3_200960:


conv1d_3_200962:
%
conv1d_4_200965:

conv1d_4_200967:
identity?? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall? conv1d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_3dense_200928dense_200930*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2005282
dense/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2005392
dropout_2/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_1_200934dense_1_200936*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2005522!
dense_1/StatefulPartitionedCall?
dropout_3/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2005632
dropout_3/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_2_200940dense_2_200942*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2005752!
dense_2/StatefulPartitionedCall?
tf.reshape_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????     2
tf.reshape_2/Reshape/shape?
tf.reshape_2/ReshapeReshape(dense_2/StatefulPartitionedCall:output:0#tf.reshape_2/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
tf.reshape_2/Reshape?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCalltf.reshape_2/Reshape:output:0conv1d_2_200947conv1d_2_200949*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv1d_2_layer_call_and_return_conditional_losses_2005982"
 conv1d_2/StatefulPartitionedCally
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_2/Mul/y?
tf.math.multiply_2/MulMul)conv1d_2/StatefulPartitionedCall:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*,
_output_shapes
:??????????
2
tf.math.multiply_2/Mulo
tf.math.add_2/Add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
tf.math.add_2/Add/y?
tf.math.add_2/AddAddV2tf.math.multiply_2/Mul:z:0tf.math.add_2/Add/y:output:0*
T0*,
_output_shapes
:??????????
2
tf.math.add_2/Add?
*tf.clip_by_value_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*tf.clip_by_value_2/clip_by_value/Minimum/y?
(tf.clip_by_value_2/clip_by_value/MinimumMinimumtf.math.add_2/Add:z:03tf.clip_by_value_2/clip_by_value/Minimum/y:output:0*
T0*,
_output_shapes
:??????????
2*
(tf.clip_by_value_2/clip_by_value/Minimum?
"tf.clip_by_value_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_2/clip_by_value/y?
 tf.clip_by_value_2/clip_by_valueMaximum,tf.clip_by_value_2/clip_by_value/Minimum:z:0+tf.clip_by_value_2/clip_by_value/y:output:0*
T0*,
_output_shapes
:??????????
2"
 tf.clip_by_value_2/clip_by_value?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_2/clip_by_value:z:0conv1d_3_200960conv1d_3_200962*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv1d_3_layer_call_and_return_conditional_losses_2006272"
 conv1d_3/StatefulPartitionedCall?
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0conv1d_4_200965conv1d_4_200967*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_2006482"
 conv1d_4/StatefulPartitionedCall?
tf.reshape_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2
tf.reshape_3/Reshape/shape?
tf.reshape_3/ReshapeReshape)conv1d_4/StatefulPartitionedCall:output:0#tf.reshape_3/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2
tf.reshape_3/Reshapey
IdentityIdentitytf.reshape_3/Reshape:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : : : 2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
'
_output_shapes
:?????????P
!
_user_specified_name	input_3
?
?
C__inference_dense_1_layer_call_and_return_conditional_losses_200552

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidg
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_dense_layer_call_fn_201292

inputs
unknown:	P?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2005282
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?

?
C__inference_dense_2_layer_call_and_return_conditional_losses_201376

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?s
?	
C__inference_model_2_layer_call_and_return_conditional_losses_201214

inputs7
$dense_matmul_readvariableop_resource:	P?4
%dense_biasadd_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?:
&dense_2_matmul_readvariableop_resource:
??6
'dense_2_biasadd_readvariableop_resource:	?J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:
6
(conv1d_2_biasadd_readvariableop_resource:
J
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:

6
(conv1d_3_biasadd_readvariableop_resource:
J
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:
6
(conv1d_4_biasadd_readvariableop_resource:
identity??conv1d_2/BiasAdd/ReadVariableOp?+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?conv1d_3/BiasAdd/ReadVariableOp?+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?conv1d_4/BiasAdd/ReadVariableOp?+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	P?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_2/dropout/Const?
dropout_2/dropout/MulMuldense/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_2/dropout/Mulz
dropout_2/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_2/dropout/Mul_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddz
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Sigmoidw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_3/dropout/Const?
dropout_3/dropout/MulMuldense_1/Sigmoid:y:0 dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_3/dropout/Mulu
dropout_3/dropout/ShapeShapedense_1/Sigmoid:y:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform?
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_3/dropout/GreaterEqual/y?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2 
dropout_3/dropout/GreaterEqual?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_3/dropout/Cast?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_3/dropout/Mul_1?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldropout_3/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAdd?
tf.reshape_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????     2
tf.reshape_2/Reshape/shape?
tf.reshape_2/ReshapeReshapedense_2/BiasAdd:output:0#tf.reshape_2/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
tf.reshape_2/Reshape?
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_2/conv1d/ExpandDims/dim?
conv1d_2/conv1d/ExpandDims
ExpandDimstf.reshape_2/Reshape:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d_2/conv1d/ExpandDims?
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
*
paddingSAME*
strides
2
conv1d_2/conv1d?
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*,
_output_shapes
:??????????
*
squeeze_dims

?????????2
conv1d_2/conv1d/Squeeze?
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp?
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
2
conv1d_2/BiasAddy
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_2/Mul/y?
tf.math.multiply_2/MulMulconv1d_2/BiasAdd:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*,
_output_shapes
:??????????
2
tf.math.multiply_2/Mulo
tf.math.add_2/Add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
tf.math.add_2/Add/y?
tf.math.add_2/AddAddV2tf.math.multiply_2/Mul:z:0tf.math.add_2/Add/y:output:0*
T0*,
_output_shapes
:??????????
2
tf.math.add_2/Add?
*tf.clip_by_value_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*tf.clip_by_value_2/clip_by_value/Minimum/y?
(tf.clip_by_value_2/clip_by_value/MinimumMinimumtf.math.add_2/Add:z:03tf.clip_by_value_2/clip_by_value/Minimum/y:output:0*
T0*,
_output_shapes
:??????????
2*
(tf.clip_by_value_2/clip_by_value/Minimum?
"tf.clip_by_value_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_2/clip_by_value/y?
 tf.clip_by_value_2/clip_by_valueMaximum,tf.clip_by_value_2/clip_by_value/Minimum:z:0+tf.clip_by_value_2/clip_by_value/y:output:0*
T0*,
_output_shapes
:??????????
2"
 tf.clip_by_value_2/clip_by_value?
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_3/conv1d/ExpandDims/dim?
conv1d_3/conv1d/ExpandDims
ExpandDims$tf.clip_by_value_2/clip_by_value:z:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
2
conv1d_3/conv1d/ExpandDims?
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:

*
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dim?
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:

2
conv1d_3/conv1d/ExpandDims_1?
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
*
paddingSAME*
strides
2
conv1d_3/conv1d?
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*,
_output_shapes
:??????????
*
squeeze_dims

?????????2
conv1d_3/conv1d/Squeeze?
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
conv1d_3/BiasAdd/ReadVariableOp?
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
2
conv1d_3/BiasAdd?
conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_4/conv1d/ExpandDims/dim?
conv1d_4/conv1d/ExpandDims
ExpandDimsconv1d_3/BiasAdd:output:0'conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
2
conv1d_4/conv1d/ExpandDims?
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02-
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_4/conv1d/ExpandDims_1/dim?
conv1d_4/conv1d/ExpandDims_1
ExpandDims3conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d_4/conv1d/ExpandDims_1?
conv1d_4/conv1dConv2D#conv1d_4/conv1d/ExpandDims:output:0%conv1d_4/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv1d_4/conv1d?
conv1d_4/conv1d/SqueezeSqueezeconv1d_4/conv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
conv1d_4/conv1d/Squeeze?
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_4/BiasAdd/ReadVariableOp?
conv1d_4/BiasAddBiasAdd conv1d_4/conv1d/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
conv1d_4/BiasAdd?
tf.reshape_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2
tf.reshape_3/Reshape/shape?
tf.reshape_3/ReshapeReshapeconv1d_4/BiasAdd:output:0#tf.reshape_3/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2
tf.reshape_3/Reshapey
IdentityIdentitytf.reshape_3/Reshape:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_4/BiasAdd/ReadVariableOp,^conv1d_4/conv1d/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : : : 2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_4/BiasAdd/ReadVariableOpconv1d_4/BiasAdd/ReadVariableOp2Z
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?`
?	
C__inference_model_2_layer_call_and_return_conditional_losses_201129

inputs7
$dense_matmul_readvariableop_resource:	P?4
%dense_biasadd_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?:
&dense_2_matmul_readvariableop_resource:
??6
'dense_2_biasadd_readvariableop_resource:	?J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:
6
(conv1d_2_biasadd_readvariableop_resource:
J
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:

6
(conv1d_3_biasadd_readvariableop_resource:
J
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:
6
(conv1d_4_biasadd_readvariableop_resource:
identity??conv1d_2/BiasAdd/ReadVariableOp?+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?conv1d_3/BiasAdd/ReadVariableOp?+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?conv1d_4/BiasAdd/ReadVariableOp?+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	P?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dropout_2/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_2/Identity?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout_2/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddz
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Sigmoid|
dropout_3/IdentityIdentitydense_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
dropout_3/Identity?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldropout_3/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAdd?
tf.reshape_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????     2
tf.reshape_2/Reshape/shape?
tf.reshape_2/ReshapeReshapedense_2/BiasAdd:output:0#tf.reshape_2/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
tf.reshape_2/Reshape?
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_2/conv1d/ExpandDims/dim?
conv1d_2/conv1d/ExpandDims
ExpandDimstf.reshape_2/Reshape:output:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d_2/conv1d/ExpandDims?
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
*
paddingSAME*
strides
2
conv1d_2/conv1d?
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*,
_output_shapes
:??????????
*
squeeze_dims

?????????2
conv1d_2/conv1d/Squeeze?
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp?
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
2
conv1d_2/BiasAddy
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_2/Mul/y?
tf.math.multiply_2/MulMulconv1d_2/BiasAdd:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*,
_output_shapes
:??????????
2
tf.math.multiply_2/Mulo
tf.math.add_2/Add/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
tf.math.add_2/Add/y?
tf.math.add_2/AddAddV2tf.math.multiply_2/Mul:z:0tf.math.add_2/Add/y:output:0*
T0*,
_output_shapes
:??????????
2
tf.math.add_2/Add?
*tf.clip_by_value_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*tf.clip_by_value_2/clip_by_value/Minimum/y?
(tf.clip_by_value_2/clip_by_value/MinimumMinimumtf.math.add_2/Add:z:03tf.clip_by_value_2/clip_by_value/Minimum/y:output:0*
T0*,
_output_shapes
:??????????
2*
(tf.clip_by_value_2/clip_by_value/Minimum?
"tf.clip_by_value_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_2/clip_by_value/y?
 tf.clip_by_value_2/clip_by_valueMaximum,tf.clip_by_value_2/clip_by_value/Minimum:z:0+tf.clip_by_value_2/clip_by_value/y:output:0*
T0*,
_output_shapes
:??????????
2"
 tf.clip_by_value_2/clip_by_value?
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_3/conv1d/ExpandDims/dim?
conv1d_3/conv1d/ExpandDims
ExpandDims$tf.clip_by_value_2/clip_by_value:z:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
2
conv1d_3/conv1d/ExpandDims?
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:

*
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dim?
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:

2
conv1d_3/conv1d/ExpandDims_1?
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????
*
paddingSAME*
strides
2
conv1d_3/conv1d?
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*,
_output_shapes
:??????????
*
squeeze_dims

?????????2
conv1d_3/conv1d/Squeeze?
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
conv1d_3/BiasAdd/ReadVariableOp?
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????
2
conv1d_3/BiasAdd?
conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_4/conv1d/ExpandDims/dim?
conv1d_4/conv1d/ExpandDims
ExpandDimsconv1d_3/BiasAdd:output:0'conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????
2
conv1d_4/conv1d/ExpandDims?
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
*
dtype02-
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_4/conv1d/ExpandDims_1/dim?
conv1d_4/conv1d/ExpandDims_1
ExpandDims3conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
2
conv1d_4/conv1d/ExpandDims_1?
conv1d_4/conv1dConv2D#conv1d_4/conv1d/ExpandDims:output:0%conv1d_4/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv1d_4/conv1d?
conv1d_4/conv1d/SqueezeSqueezeconv1d_4/conv1d:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????2
conv1d_4/conv1d/Squeeze?
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_4/BiasAdd/ReadVariableOp?
conv1d_4/BiasAddBiasAdd conv1d_4/conv1d/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
conv1d_4/BiasAdd?
tf.reshape_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????  2
tf.reshape_3/Reshape/shape?
tf.reshape_3/ReshapeReshapeconv1d_4/BiasAdd:output:0#tf.reshape_3/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2
tf.reshape_3/Reshapey
IdentityIdentitytf.reshape_3/Reshape:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_3/BiasAdd/ReadVariableOp,^conv1d_3/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_4/BiasAdd/ReadVariableOp,^conv1d_4/conv1d/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : : : 2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_3/BiasAdd/ReadVariableOpconv1d_3/BiasAdd/ReadVariableOp2Z
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_4/BiasAdd/ReadVariableOpconv1d_4/BiasAdd/ReadVariableOp2Z
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
C__inference_dense_1_layer_call_and_return_conditional_losses_201330

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidg
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_model_2_layer_call_fn_200925
input_3
unknown:	P?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:

	unknown_6:

	unknown_7:


	unknown_8:

	unknown_9:


unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_2008692
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????P
!
_user_specified_name	input_3
?
?
$__inference_signature_wrapper_201058
input_3
unknown:	P?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:	?
	unknown_5:

	unknown_6:

	unknown_7:


	unknown_8:

	unknown_9:


unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? **
f%R#
!__inference__wrapped_model_2005102
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????P: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????P
!
_user_specified_name	input_3"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_30
serving_default_input_3:0?????????PA
tf.reshape_31
StatefulPartitionedCall:0??????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
	optimizer

signatures
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_network
D
#_self_saveable_object_factories"
_tf_keras_input_layer
?

kernel
bias
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
#_self_saveable_object_factories
trainable_variables
 	variables
!regularization_losses
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

#kernel
$bias
#%_self_saveable_object_factories
&trainable_variables
'	variables
(regularization_losses
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
#*_self_saveable_object_factories
+trainable_variables
,	variables
-regularization_losses
.	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

/kernel
0bias
#1_self_saveable_object_factories
2trainable_variables
3	variables
4regularization_losses
5	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
M
#6_self_saveable_object_factories
7	keras_api"
_tf_keras_layer
?

8kernel
9bias
#:_self_saveable_object_factories
;trainable_variables
<	variables
=regularization_losses
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
M
#?_self_saveable_object_factories
@	keras_api"
_tf_keras_layer
M
#A_self_saveable_object_factories
B	keras_api"
_tf_keras_layer
M
#C_self_saveable_object_factories
D	keras_api"
_tf_keras_layer
?

Ekernel
Fbias
#G_self_saveable_object_factories
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Lkernel
Mbias
#N_self_saveable_object_factories
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
M
#S_self_saveable_object_factories
T	keras_api"
_tf_keras_layer
?
Uiter

Vbeta_1

Wbeta_2
	Xdecay
Ylearning_ratem?m?#m?$m?/m?0m?8m?9m?Em?Fm?Lm?Mm?v?v?#v?$v?/v?0v?8v?9v?Ev?Fv?Lv?Mv?"
	optimizer
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
v
0
1
#2
$3
/4
05
86
97
E8
F9
L10
M11"
trackable_list_wrapper
v
0
1
#2
$3
/4
05
86
97
E8
F9
L10
M11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Zmetrics
trainable_variables
	variables
[layer_metrics
\non_trainable_variables
]layer_regularization_losses
regularization_losses

^layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
:	P?2dense/kernel
:?2
dense/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
_metrics
trainable_variables
	variables
`layer_metrics
anon_trainable_variables
blayer_regularization_losses
regularization_losses

clayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
dmetrics
trainable_variables
 	variables
elayer_metrics
fnon_trainable_variables
glayer_regularization_losses
!regularization_losses

hlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_1/kernel
:?2dense_1/bias
 "
trackable_dict_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
imetrics
&trainable_variables
'	variables
jlayer_metrics
knon_trainable_variables
llayer_regularization_losses
(regularization_losses

mlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
nmetrics
+trainable_variables
,	variables
olayer_metrics
pnon_trainable_variables
qlayer_regularization_losses
-regularization_losses

rlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_2/kernel
:?2dense_2/bias
 "
trackable_dict_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
smetrics
2trainable_variables
3	variables
tlayer_metrics
unon_trainable_variables
vlayer_regularization_losses
4regularization_losses

wlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
"
_generic_user_object
%:#
2conv1d_2/kernel
:
2conv1d_2/bias
 "
trackable_dict_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?
xmetrics
;trainable_variables
<	variables
ylayer_metrics
znon_trainable_variables
{layer_regularization_losses
=regularization_losses

|layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
"
_generic_user_object
%:#

2conv1d_3/kernel
:
2conv1d_3/bias
 "
trackable_dict_wrapper
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
}metrics
Htrainable_variables
I	variables
~layer_metrics
non_trainable_variables
 ?layer_regularization_losses
Jregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#
2conv1d_4/kernel
:2conv1d_4/bias
 "
trackable_dict_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
Otrainable_variables
P	variables
?layer_metrics
?non_trainable_variables
 ?layer_regularization_losses
Qregularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
$:"	P?2Adam/dense/kernel/m
:?2Adam/dense/bias/m
':%
??2Adam/dense_1/kernel/m
 :?2Adam/dense_1/bias/m
':%
??2Adam/dense_2/kernel/m
 :?2Adam/dense_2/bias/m
*:(
2Adam/conv1d_2/kernel/m
 :
2Adam/conv1d_2/bias/m
*:(

2Adam/conv1d_3/kernel/m
 :
2Adam/conv1d_3/bias/m
*:(
2Adam/conv1d_4/kernel/m
 :2Adam/conv1d_4/bias/m
$:"	P?2Adam/dense/kernel/v
:?2Adam/dense/bias/v
':%
??2Adam/dense_1/kernel/v
 :?2Adam/dense_1/bias/v
':%
??2Adam/dense_2/kernel/v
 :?2Adam/dense_2/bias/v
*:(
2Adam/conv1d_2/kernel/v
 :
2Adam/conv1d_2/bias/v
*:(

2Adam/conv1d_3/kernel/v
 :
2Adam/conv1d_3/bias/v
*:(
2Adam/conv1d_4/kernel/v
 :2Adam/conv1d_4/bias/v
?B?
!__inference__wrapped_model_200510input_3"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_model_2_layer_call_and_return_conditional_losses_201129
C__inference_model_2_layer_call_and_return_conditional_losses_201214
C__inference_model_2_layer_call_and_return_conditional_losses_200973
C__inference_model_2_layer_call_and_return_conditional_losses_201021?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_model_2_layer_call_fn_200684
(__inference_model_2_layer_call_fn_201243
(__inference_model_2_layer_call_fn_201272
(__inference_model_2_layer_call_fn_200925?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_dense_layer_call_and_return_conditional_losses_201283?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_layer_call_fn_201292?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dropout_2_layer_call_and_return_conditional_losses_201297
E__inference_dropout_2_layer_call_and_return_conditional_losses_201309?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_2_layer_call_fn_201314
*__inference_dropout_2_layer_call_fn_201319?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dense_1_layer_call_and_return_conditional_losses_201330?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_1_layer_call_fn_201339?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dropout_3_layer_call_and_return_conditional_losses_201344
E__inference_dropout_3_layer_call_and_return_conditional_losses_201356?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_3_layer_call_fn_201361
*__inference_dropout_3_layer_call_fn_201366?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dense_2_layer_call_and_return_conditional_losses_201376?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_2_layer_call_fn_201385?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv1d_2_layer_call_and_return_conditional_losses_201400?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv1d_2_layer_call_fn_201409?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv1d_3_layer_call_and_return_conditional_losses_201424?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv1d_3_layer_call_fn_201433?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv1d_4_layer_call_and_return_conditional_losses_201448?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv1d_4_layer_call_fn_201457?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_201058input_3"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_200510~#$/089EFLM0?-
&?#
!?
input_3?????????P
? "<?9
7
tf.reshape_3'?$
tf.reshape_3???????????
D__inference_conv1d_2_layer_call_and_return_conditional_losses_201400f894?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????

? ?
)__inference_conv1d_2_layer_call_fn_201409Y894?1
*?'
%?"
inputs??????????
? "???????????
?
D__inference_conv1d_3_layer_call_and_return_conditional_losses_201424fEF4?1
*?'
%?"
inputs??????????

? "*?'
 ?
0??????????

? ?
)__inference_conv1d_3_layer_call_fn_201433YEF4?1
*?'
%?"
inputs??????????

? "???????????
?
D__inference_conv1d_4_layer_call_and_return_conditional_losses_201448fLM4?1
*?'
%?"
inputs??????????

? "*?'
 ?
0??????????
? ?
)__inference_conv1d_4_layer_call_fn_201457YLM4?1
*?'
%?"
inputs??????????

? "????????????
C__inference_dense_1_layer_call_and_return_conditional_losses_201330^#$0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_dense_1_layer_call_fn_201339Q#$0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dense_2_layer_call_and_return_conditional_losses_201376^/00?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_dense_2_layer_call_fn_201385Q/00?-
&?#
!?
inputs??????????
? "????????????
A__inference_dense_layer_call_and_return_conditional_losses_201283]/?,
%?"
 ?
inputs?????????P
? "&?#
?
0??????????
? z
&__inference_dense_layer_call_fn_201292P/?,
%?"
 ?
inputs?????????P
? "????????????
E__inference_dropout_2_layer_call_and_return_conditional_losses_201297^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
E__inference_dropout_2_layer_call_and_return_conditional_losses_201309^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? 
*__inference_dropout_2_layer_call_fn_201314Q4?1
*?'
!?
inputs??????????
p 
? "???????????
*__inference_dropout_2_layer_call_fn_201319Q4?1
*?'
!?
inputs??????????
p
? "????????????
E__inference_dropout_3_layer_call_and_return_conditional_losses_201344^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
E__inference_dropout_3_layer_call_and_return_conditional_losses_201356^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? 
*__inference_dropout_3_layer_call_fn_201361Q4?1
*?'
!?
inputs??????????
p 
? "???????????
*__inference_dropout_3_layer_call_fn_201366Q4?1
*?'
!?
inputs??????????
p
? "????????????
C__inference_model_2_layer_call_and_return_conditional_losses_200973p#$/089EFLM8?5
.?+
!?
input_3?????????P
p 

 
? "&?#
?
0??????????
? ?
C__inference_model_2_layer_call_and_return_conditional_losses_201021p#$/089EFLM8?5
.?+
!?
input_3?????????P
p

 
? "&?#
?
0??????????
? ?
C__inference_model_2_layer_call_and_return_conditional_losses_201129o#$/089EFLM7?4
-?*
 ?
inputs?????????P
p 

 
? "&?#
?
0??????????
? ?
C__inference_model_2_layer_call_and_return_conditional_losses_201214o#$/089EFLM7?4
-?*
 ?
inputs?????????P
p

 
? "&?#
?
0??????????
? ?
(__inference_model_2_layer_call_fn_200684c#$/089EFLM8?5
.?+
!?
input_3?????????P
p 

 
? "????????????
(__inference_model_2_layer_call_fn_200925c#$/089EFLM8?5
.?+
!?
input_3?????????P
p

 
? "????????????
(__inference_model_2_layer_call_fn_201243b#$/089EFLM7?4
-?*
 ?
inputs?????????P
p 

 
? "????????????
(__inference_model_2_layer_call_fn_201272b#$/089EFLM7?4
-?*
 ?
inputs?????????P
p

 
? "????????????
$__inference_signature_wrapper_201058?#$/089EFLM;?8
? 
1?.
,
input_3!?
input_3?????????P"<?9
7
tf.reshape_3'?$
tf.reshape_3??????????