??
??
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
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
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
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
?
embedding_8/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?E?*'
shared_nameembedding_8/embeddings
?
*embedding_8/embeddings/Read/ReadVariableOpReadVariableOpembedding_8/embeddings* 
_output_shapes
:
?E?*
dtype0
?
conv1d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?e*!
shared_nameconv1d_15/kernel
z
$conv1d_15/kernel/Read/ReadVariableOpReadVariableOpconv1d_15/kernel*#
_output_shapes
:?e*
dtype0
t
conv1d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:e*
shared_nameconv1d_15/bias
m
"conv1d_15/bias/Read/ReadVariableOpReadVariableOpconv1d_15/bias*
_output_shapes
:e*
dtype0
?
conv1d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*!
shared_nameconv1d_16/kernel
y
$conv1d_16/kernel/Read/ReadVariableOpReadVariableOpconv1d_16/kernel*"
_output_shapes
:P*
dtype0
t
conv1d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_nameconv1d_16/bias
m
"conv1d_16/bias/Read/ReadVariableOpReadVariableOpconv1d_16/bias*
_output_shapes
:P*
dtype0
?
conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*!
shared_nameconv2d_14/kernel
}
$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*&
_output_shapes
:2*
dtype0
t
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_nameconv2d_14/bias
m
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes
:2*
dtype0
?
conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
:2*
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
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

NoOpNoOp
?&
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?&
value?%B?% B?%
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
b

embeddings
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api

	keras_api
R
trainable_variables
regularization_losses
 	variables
!	keras_api
h

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api

(	keras_api
h

)kernel
*bias
+trainable_variables
,regularization_losses
-	variables
.	keras_api
R
/trainable_variables
0regularization_losses
1	variables
2	keras_api
h

3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api

9	keras_api
@
:iter

;beta_1

<beta_2
	=decay
>learning_rate
?
0
1
2
"3
#4
)5
*6
37
48
 
?
0
1
2
"3
#4
)5
*6
37
48
?

?layers
trainable_variables
@layer_metrics
Anon_trainable_variables
regularization_losses
Bmetrics
	variables
Clayer_regularization_losses
 
fd
VARIABLE_VALUEembedding_8/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
?

Dlayers
trainable_variables
Elayer_metrics
Fnon_trainable_variables
regularization_losses
Gmetrics
	variables
Hlayer_regularization_losses
\Z
VARIABLE_VALUEconv1d_15/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_15/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

Ilayers
trainable_variables
Jlayer_metrics
Knon_trainable_variables
regularization_losses
Lmetrics
	variables
Mlayer_regularization_losses
 
 
 
 
?

Nlayers
trainable_variables
Olayer_metrics
Pnon_trainable_variables
regularization_losses
Qmetrics
 	variables
Rlayer_regularization_losses
\Z
VARIABLE_VALUEconv1d_16/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv1d_16/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
?

Slayers
$trainable_variables
Tlayer_metrics
Unon_trainable_variables
%regularization_losses
Vmetrics
&	variables
Wlayer_regularization_losses
 
\Z
VARIABLE_VALUEconv2d_14/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_14/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1
 

)0
*1
?

Xlayers
+trainable_variables
Ylayer_metrics
Znon_trainable_variables
,regularization_losses
[metrics
-	variables
\layer_regularization_losses
 
 
 
?

]layers
/trainable_variables
^layer_metrics
_non_trainable_variables
0regularization_losses
`metrics
1	variables
alayer_regularization_losses
\Z
VARIABLE_VALUEconv2d_15/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_15/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41
?

blayers
5trainable_variables
clayer_metrics
dnon_trainable_variables
6regularization_losses
emetrics
7	variables
flayer_regularization_losses
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
N
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
 
 

g0
h1
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
4
	itotal
	jcount
k	variables
l	keras_api
D
	mtotal
	ncount
o
_fn_kwargs
p	variables
q	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

i0
j1

k	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

m0
n1

p	variables
{
serving_default_input_10Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_10embedding_8/embeddingsconv1d_15/kernelconv1d_15/biasconv1d_16/kernelconv1d_16/biasconv2d_14/kernelconv2d_14/biasconv2d_15/kernelconv2d_15/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????eP*+
_read_only_resource_inputs
		*2
config_proto" 

CPU

GPU2*0,1J 8? *-
f(R&
$__inference_signature_wrapper_434975
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_8/embeddings/Read/ReadVariableOp$conv1d_15/kernel/Read/ReadVariableOp"conv1d_15/bias/Read/ReadVariableOp$conv1d_16/kernel/Read/ReadVariableOp"conv1d_16/bias/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2	*
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
__inference__traced_save_435379
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_8/embeddingsconv1d_15/kernelconv1d_15/biasconv1d_16/kernelconv1d_16/biasconv2d_14/kernelconv2d_14/biasconv2d_15/kernelconv2d_15/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1*
Tin
2*
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
"__inference__traced_restore_435443??
?

?
&__inference_model_layer_call_fn_434676
input_10
unknown:
?E? 
	unknown_0:?e
	unknown_1:e
	unknown_2:P
	unknown_3:P#
	unknown_4:2
	unknown_5:2#
	unknown_6:2
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????eP*+
_read_only_resource_inputs
		*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4346552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????eP2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_10
?
?
E__inference_conv1d_15_layer_call_and_return_conditional_losses_435175

inputsB
+conv1d_expanddims_1_readvariableop_resource:?e-
biasadd_readvariableop_resource:e
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
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?e*
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
T0*'
_output_shapes
:?e2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????e*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????e*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:e*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????e2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????e2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_14_layer_call_and_return_conditional_losses_434623

inputs8
conv2d_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP2*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP22	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????eP22	
Sigmoidn
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????eP22

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????eP: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????eP
 
_user_specified_nameinputs
?/
?
A__inference_model_layer_call_and_return_conditional_losses_434944
input_10&
embedding_8_434912:
?E?'
conv1d_15_434915:?e
conv1d_15_434917:e&
conv1d_16_434923:P
conv1d_16_434925:P*
conv2d_14_434930:2
conv2d_14_434932:2*
conv2d_15_434936:2
conv2d_15_434938:
identity??!conv1d_15/StatefulPartitionedCall?!conv1d_16/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?#embedding_8/StatefulPartitionedCall?
#embedding_8/StatefulPartitionedCallStatefulPartitionedCallinput_10embedding_8_434912*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_embedding_8_layer_call_and_return_conditional_losses_4345542%
#embedding_8/StatefulPartitionedCall?
!conv1d_15/StatefulPartitionedCallStatefulPartitionedCall,embedding_8/StatefulPartitionedCall:output:0conv1d_15_434915conv1d_15_434917*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????e*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv1d_15_layer_call_and_return_conditional_losses_4345732#
!conv1d_15/StatefulPartitionedCall?
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2)
'tf.compat.v1.transpose_1/transpose/perm?
"tf.compat.v1.transpose_1/transpose	Transpose*conv1d_15/StatefulPartitionedCall:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????e2$
"tf.compat.v1.transpose_1/transpose?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall&tf.compat.v1.transpose_1/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????e* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_4347492#
!dropout_7/StatefulPartitionedCall?
!conv1d_16/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0conv1d_16_434923conv1d_16_434925*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????eP*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv1d_16_layer_call_and_return_conditional_losses_4346042#
!conv1d_16/StatefulPartitionedCall?
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.expand_dims/ExpandDims/dim?
tf.expand_dims/ExpandDims
ExpandDims*conv1d_16/StatefulPartitionedCall:output:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????eP2
tf.expand_dims/ExpandDims?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall"tf.expand_dims/ExpandDims:output:0conv2d_14_434930conv2d_14_434932*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????eP2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_4346232#
!conv2d_14/StatefulPartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????eP2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_4347062#
!dropout_8/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0conv2d_15_434936conv2d_15_434938*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????eP*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_4346462#
!conv2d_15/StatefulPartitionedCall?
tf.reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????e   P   2
tf.reshape/Reshape/shape?
tf.reshape/ReshapeReshape*conv2d_15/StatefulPartitionedCall:output:0!tf.reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????eP2
tf.reshape/Reshapez
IdentityIdentitytf.reshape/Reshape:output:0^NoOp*
T0*+
_output_shapes
:?????????eP2

Identity?
NoOpNoOp"^conv1d_15/StatefulPartitionedCall"^conv1d_16/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall$^embedding_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 2F
!conv1d_15/StatefulPartitionedCall!conv1d_15/StatefulPartitionedCall2F
!conv1d_16/StatefulPartitionedCall!conv1d_16/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2J
#embedding_8/StatefulPartitionedCall#embedding_8/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_10
?

?
G__inference_embedding_8_layer_call_and_return_conditional_losses_435153

inputs+
embedding_lookup_435147:
?E?
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_435147Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0**
_class 
loc:@embedding_lookup/435147*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@embedding_lookup/435147*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_conv1d_16_layer_call_and_return_conditional_losses_435227

inputsA
+conv1d_expanddims_1_readvariableop_resource:P-
biasadd_readvariableop_resource:P
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
T0*/
_output_shapes
:?????????e2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:P*
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
:P2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????eP*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????eP*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????eP2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????eP2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????eP2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????e: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????e
 
_user_specified_nameinputs
?
?
E__inference_conv2d_14_layer_call_and_return_conditional_losses_435247

inputs8
conv2d_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP2*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP22	
BiasAddi
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????eP22	
Sigmoidn
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????eP22

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????eP: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????eP
 
_user_specified_nameinputs
?U
?
!__inference__wrapped_model_434537
input_10=
)model_embedding_8_embedding_lookup_434487:
?E?R
;model_conv1d_15_conv1d_expanddims_1_readvariableop_resource:?e=
/model_conv1d_15_biasadd_readvariableop_resource:eQ
;model_conv1d_16_conv1d_expanddims_1_readvariableop_resource:P=
/model_conv1d_16_biasadd_readvariableop_resource:PH
.model_conv2d_14_conv2d_readvariableop_resource:2=
/model_conv2d_14_biasadd_readvariableop_resource:2H
.model_conv2d_15_conv2d_readvariableop_resource:2=
/model_conv2d_15_biasadd_readvariableop_resource:
identity??&model/conv1d_15/BiasAdd/ReadVariableOp?2model/conv1d_15/conv1d/ExpandDims_1/ReadVariableOp?&model/conv1d_16/BiasAdd/ReadVariableOp?2model/conv1d_16/conv1d/ExpandDims_1/ReadVariableOp?&model/conv2d_14/BiasAdd/ReadVariableOp?%model/conv2d_14/Conv2D/ReadVariableOp?&model/conv2d_15/BiasAdd/ReadVariableOp?%model/conv2d_15/Conv2D/ReadVariableOp?"model/embedding_8/embedding_lookup?
model/embedding_8/CastCastinput_10*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model/embedding_8/Cast?
"model/embedding_8/embedding_lookupResourceGather)model_embedding_8_embedding_lookup_434487model/embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*<
_class2
0.loc:@model/embedding_8/embedding_lookup/434487*,
_output_shapes
:??????????*
dtype02$
"model/embedding_8/embedding_lookup?
+model/embedding_8/embedding_lookup/IdentityIdentity+model/embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*<
_class2
0.loc:@model/embedding_8/embedding_lookup/434487*,
_output_shapes
:??????????2-
+model/embedding_8/embedding_lookup/Identity?
-model/embedding_8/embedding_lookup/Identity_1Identity4model/embedding_8/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2/
-model/embedding_8/embedding_lookup/Identity_1?
%model/conv1d_15/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%model/conv1d_15/conv1d/ExpandDims/dim?
!model/conv1d_15/conv1d/ExpandDims
ExpandDims6model/embedding_8/embedding_lookup/Identity_1:output:0.model/conv1d_15/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2#
!model/conv1d_15/conv1d/ExpandDims?
2model/conv1d_15/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;model_conv1d_15_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?e*
dtype024
2model/conv1d_15/conv1d/ExpandDims_1/ReadVariableOp?
'model/conv1d_15/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model/conv1d_15/conv1d/ExpandDims_1/dim?
#model/conv1d_15/conv1d/ExpandDims_1
ExpandDims:model/conv1d_15/conv1d/ExpandDims_1/ReadVariableOp:value:00model/conv1d_15/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?e2%
#model/conv1d_15/conv1d/ExpandDims_1?
model/conv1d_15/conv1dConv2D*model/conv1d_15/conv1d/ExpandDims:output:0,model/conv1d_15/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????e*
paddingSAME*
strides
2
model/conv1d_15/conv1d?
model/conv1d_15/conv1d/SqueezeSqueezemodel/conv1d_15/conv1d:output:0*
T0*+
_output_shapes
:?????????e*
squeeze_dims

?????????2 
model/conv1d_15/conv1d/Squeeze?
&model/conv1d_15/BiasAdd/ReadVariableOpReadVariableOp/model_conv1d_15_biasadd_readvariableop_resource*
_output_shapes
:e*
dtype02(
&model/conv1d_15/BiasAdd/ReadVariableOp?
model/conv1d_15/BiasAddBiasAdd'model/conv1d_15/conv1d/Squeeze:output:0.model/conv1d_15/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????e2
model/conv1d_15/BiasAdd?
-model/tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2/
-model/tf.compat.v1.transpose_1/transpose/perm?
(model/tf.compat.v1.transpose_1/transpose	Transpose model/conv1d_15/BiasAdd:output:06model/tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????e2*
(model/tf.compat.v1.transpose_1/transpose?
model/dropout_7/IdentityIdentity,model/tf.compat.v1.transpose_1/transpose:y:0*
T0*+
_output_shapes
:?????????e2
model/dropout_7/Identity?
%model/conv1d_16/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%model/conv1d_16/conv1d/ExpandDims/dim?
!model/conv1d_16/conv1d/ExpandDims
ExpandDims!model/dropout_7/Identity:output:0.model/conv1d_16/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????e2#
!model/conv1d_16/conv1d/ExpandDims?
2model/conv1d_16/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;model_conv1d_16_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:P*
dtype024
2model/conv1d_16/conv1d/ExpandDims_1/ReadVariableOp?
'model/conv1d_16/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model/conv1d_16/conv1d/ExpandDims_1/dim?
#model/conv1d_16/conv1d/ExpandDims_1
ExpandDims:model/conv1d_16/conv1d/ExpandDims_1/ReadVariableOp:value:00model/conv1d_16/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:P2%
#model/conv1d_16/conv1d/ExpandDims_1?
model/conv1d_16/conv1dConv2D*model/conv1d_16/conv1d/ExpandDims:output:0,model/conv1d_16/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????eP*
paddingSAME*
strides
2
model/conv1d_16/conv1d?
model/conv1d_16/conv1d/SqueezeSqueezemodel/conv1d_16/conv1d:output:0*
T0*+
_output_shapes
:?????????eP*
squeeze_dims

?????????2 
model/conv1d_16/conv1d/Squeeze?
&model/conv1d_16/BiasAdd/ReadVariableOpReadVariableOp/model_conv1d_16_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02(
&model/conv1d_16/BiasAdd/ReadVariableOp?
model/conv1d_16/BiasAddBiasAdd'model/conv1d_16/conv1d/Squeeze:output:0.model/conv1d_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????eP2
model/conv1d_16/BiasAdd?
model/conv1d_16/ReluRelu model/conv1d_16/BiasAdd:output:0*
T0*+
_output_shapes
:?????????eP2
model/conv1d_16/Relu?
#model/tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#model/tf.expand_dims/ExpandDims/dim?
model/tf.expand_dims/ExpandDims
ExpandDims"model/conv1d_16/Relu:activations:0,model/tf.expand_dims/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????eP2!
model/tf.expand_dims/ExpandDims?
%model/conv2d_14/Conv2D/ReadVariableOpReadVariableOp.model_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02'
%model/conv2d_14/Conv2D/ReadVariableOp?
model/conv2d_14/Conv2DConv2D(model/tf.expand_dims/ExpandDims:output:0-model/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP2*
paddingSAME*
strides
2
model/conv2d_14/Conv2D?
&model/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp/model_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02(
&model/conv2d_14/BiasAdd/ReadVariableOp?
model/conv2d_14/BiasAddBiasAddmodel/conv2d_14/Conv2D:output:0.model/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP22
model/conv2d_14/BiasAdd?
model/conv2d_14/SigmoidSigmoid model/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????eP22
model/conv2d_14/Sigmoid?
model/dropout_8/IdentityIdentitymodel/conv2d_14/Sigmoid:y:0*
T0*/
_output_shapes
:?????????eP22
model/dropout_8/Identity?
%model/conv2d_15/Conv2D/ReadVariableOpReadVariableOp.model_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02'
%model/conv2d_15/Conv2D/ReadVariableOp?
model/conv2d_15/Conv2DConv2D!model/dropout_8/Identity:output:0-model/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP*
paddingSAME*
strides
2
model/conv2d_15/Conv2D?
&model/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp/model_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model/conv2d_15/BiasAdd/ReadVariableOp?
model/conv2d_15/BiasAddBiasAddmodel/conv2d_15/Conv2D:output:0.model/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP2
model/conv2d_15/BiasAdd?
model/tf.reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????e   P   2 
model/tf.reshape/Reshape/shape?
model/tf.reshape/ReshapeReshape model/conv2d_15/BiasAdd:output:0'model/tf.reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????eP2
model/tf.reshape/Reshape?
IdentityIdentity!model/tf.reshape/Reshape:output:0^NoOp*
T0*+
_output_shapes
:?????????eP2

Identity?
NoOpNoOp'^model/conv1d_15/BiasAdd/ReadVariableOp3^model/conv1d_15/conv1d/ExpandDims_1/ReadVariableOp'^model/conv1d_16/BiasAdd/ReadVariableOp3^model/conv1d_16/conv1d/ExpandDims_1/ReadVariableOp'^model/conv2d_14/BiasAdd/ReadVariableOp&^model/conv2d_14/Conv2D/ReadVariableOp'^model/conv2d_15/BiasAdd/ReadVariableOp&^model/conv2d_15/Conv2D/ReadVariableOp#^model/embedding_8/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 2P
&model/conv1d_15/BiasAdd/ReadVariableOp&model/conv1d_15/BiasAdd/ReadVariableOp2h
2model/conv1d_15/conv1d/ExpandDims_1/ReadVariableOp2model/conv1d_15/conv1d/ExpandDims_1/ReadVariableOp2P
&model/conv1d_16/BiasAdd/ReadVariableOp&model/conv1d_16/BiasAdd/ReadVariableOp2h
2model/conv1d_16/conv1d/ExpandDims_1/ReadVariableOp2model/conv1d_16/conv1d/ExpandDims_1/ReadVariableOp2P
&model/conv2d_14/BiasAdd/ReadVariableOp&model/conv2d_14/BiasAdd/ReadVariableOp2N
%model/conv2d_14/Conv2D/ReadVariableOp%model/conv2d_14/Conv2D/ReadVariableOp2P
&model/conv2d_15/BiasAdd/ReadVariableOp&model/conv2d_15/BiasAdd/ReadVariableOp2N
%model/conv2d_15/Conv2D/ReadVariableOp%model/conv2d_15/Conv2D/ReadVariableOp2H
"model/embedding_8/embedding_lookup"model/embedding_8/embedding_lookup:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_10
?,
?
A__inference_model_layer_call_and_return_conditional_losses_434909
input_10&
embedding_8_434877:
?E?'
conv1d_15_434880:?e
conv1d_15_434882:e&
conv1d_16_434888:P
conv1d_16_434890:P*
conv2d_14_434895:2
conv2d_14_434897:2*
conv2d_15_434901:2
conv2d_15_434903:
identity??!conv1d_15/StatefulPartitionedCall?!conv1d_16/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?#embedding_8/StatefulPartitionedCall?
#embedding_8/StatefulPartitionedCallStatefulPartitionedCallinput_10embedding_8_434877*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_embedding_8_layer_call_and_return_conditional_losses_4345542%
#embedding_8/StatefulPartitionedCall?
!conv1d_15/StatefulPartitionedCallStatefulPartitionedCall,embedding_8/StatefulPartitionedCall:output:0conv1d_15_434880conv1d_15_434882*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????e*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv1d_15_layer_call_and_return_conditional_losses_4345732#
!conv1d_15/StatefulPartitionedCall?
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2)
'tf.compat.v1.transpose_1/transpose/perm?
"tf.compat.v1.transpose_1/transpose	Transpose*conv1d_15/StatefulPartitionedCall:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????e2$
"tf.compat.v1.transpose_1/transpose?
dropout_7/PartitionedCallPartitionedCall&tf.compat.v1.transpose_1/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????e* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_4345862
dropout_7/PartitionedCall?
!conv1d_16/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0conv1d_16_434888conv1d_16_434890*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????eP*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv1d_16_layer_call_and_return_conditional_losses_4346042#
!conv1d_16/StatefulPartitionedCall?
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.expand_dims/ExpandDims/dim?
tf.expand_dims/ExpandDims
ExpandDims*conv1d_16/StatefulPartitionedCall:output:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????eP2
tf.expand_dims/ExpandDims?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall"tf.expand_dims/ExpandDims:output:0conv2d_14_434895conv2d_14_434897*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????eP2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_4346232#
!conv2d_14/StatefulPartitionedCall?
dropout_8/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????eP2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_4346342
dropout_8/PartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0conv2d_15_434901conv2d_15_434903*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????eP*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_4346462#
!conv2d_15/StatefulPartitionedCall?
tf.reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????e   P   2
tf.reshape/Reshape/shape?
tf.reshape/ReshapeReshape*conv2d_15/StatefulPartitionedCall:output:0!tf.reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????eP2
tf.reshape/Reshapez
IdentityIdentitytf.reshape/Reshape:output:0^NoOp*
T0*+
_output_shapes
:?????????eP2

Identity?
NoOpNoOp"^conv1d_15/StatefulPartitionedCall"^conv1d_16/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall$^embedding_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 2F
!conv1d_15/StatefulPartitionedCall!conv1d_15/StatefulPartitionedCall2F
!conv1d_16/StatefulPartitionedCall!conv1d_16/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2J
#embedding_8/StatefulPartitionedCall#embedding_8/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_10
?-
?
__inference__traced_save_435379
file_prefix5
1savev2_embedding_8_embeddings_read_readvariableop/
+savev2_conv1d_15_kernel_read_readvariableop-
)savev2_conv1d_15_bias_read_readvariableop/
+savev2_conv1d_16_kernel_read_readvariableop-
)savev2_conv1d_16_bias_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_8_embeddings_read_readvariableop+savev2_conv1d_15_kernel_read_readvariableop)savev2_conv1d_15_bias_read_readvariableop+savev2_conv1d_16_kernel_read_readvariableop)savev2_conv1d_16_bias_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
2	2
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

identity_1Identity_1:output:0*?
_input_shapes}
{: :
?E?:?e:e:P:P:2:2:2:: : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
?E?:)%
#
_output_shapes
:?e: 

_output_shapes
:e:($
"
_output_shapes
:P: 

_output_shapes
:P:,(
&
_output_shapes
:2: 

_output_shapes
:2:,(
&
_output_shapes
:2: 	

_output_shapes
::


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: 
?
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_434634

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????eP22

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????eP22

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????eP2:W S
/
_output_shapes
:?????????eP2
 
_user_specified_nameinputs
?
d
E__inference_dropout_7_layer_call_and_return_conditional_losses_435201

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????e2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????e*
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
T0*+
_output_shapes
:?????????e2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????e2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????e2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????e2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????e:S O
+
_output_shapes
:?????????e
 
_user_specified_nameinputs
?/
?
A__inference_model_layer_call_and_return_conditional_losses_434830

inputs&
embedding_8_434798:
?E?'
conv1d_15_434801:?e
conv1d_15_434803:e&
conv1d_16_434809:P
conv1d_16_434811:P*
conv2d_14_434816:2
conv2d_14_434818:2*
conv2d_15_434822:2
conv2d_15_434824:
identity??!conv1d_15/StatefulPartitionedCall?!conv1d_16/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?#embedding_8/StatefulPartitionedCall?
#embedding_8/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_8_434798*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_embedding_8_layer_call_and_return_conditional_losses_4345542%
#embedding_8/StatefulPartitionedCall?
!conv1d_15/StatefulPartitionedCallStatefulPartitionedCall,embedding_8/StatefulPartitionedCall:output:0conv1d_15_434801conv1d_15_434803*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????e*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv1d_15_layer_call_and_return_conditional_losses_4345732#
!conv1d_15/StatefulPartitionedCall?
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2)
'tf.compat.v1.transpose_1/transpose/perm?
"tf.compat.v1.transpose_1/transpose	Transpose*conv1d_15/StatefulPartitionedCall:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????e2$
"tf.compat.v1.transpose_1/transpose?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall&tf.compat.v1.transpose_1/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????e* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_4347492#
!dropout_7/StatefulPartitionedCall?
!conv1d_16/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0conv1d_16_434809conv1d_16_434811*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????eP*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv1d_16_layer_call_and_return_conditional_losses_4346042#
!conv1d_16/StatefulPartitionedCall?
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.expand_dims/ExpandDims/dim?
tf.expand_dims/ExpandDims
ExpandDims*conv1d_16/StatefulPartitionedCall:output:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????eP2
tf.expand_dims/ExpandDims?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall"tf.expand_dims/ExpandDims:output:0conv2d_14_434816conv2d_14_434818*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????eP2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_4346232#
!conv2d_14/StatefulPartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????eP2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_4347062#
!dropout_8/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0conv2d_15_434822conv2d_15_434824*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????eP*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_4346462#
!conv2d_15/StatefulPartitionedCall?
tf.reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????e   P   2
tf.reshape/Reshape/shape?
tf.reshape/ReshapeReshape*conv2d_15/StatefulPartitionedCall:output:0!tf.reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????eP2
tf.reshape/Reshapez
IdentityIdentitytf.reshape/Reshape:output:0^NoOp*
T0*+
_output_shapes
:?????????eP2

Identity?
NoOpNoOp"^conv1d_15/StatefulPartitionedCall"^conv1d_16/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall$^embedding_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 2F
!conv1d_15/StatefulPartitionedCall!conv1d_15/StatefulPartitionedCall2F
!conv1d_16/StatefulPartitionedCall!conv1d_16/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2J
#embedding_8/StatefulPartitionedCall#embedding_8/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_15_layer_call_and_return_conditional_losses_434646

inputs8
conv2d_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????eP2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????eP2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????eP2
 
_user_specified_nameinputs
?
d
E__inference_dropout_8_layer_call_and_return_conditional_losses_434706

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????eP22
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????eP2*
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
T0*/
_output_shapes
:?????????eP22
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????eP22
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????eP22
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????eP22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????eP2:W S
/
_output_shapes
:?????????eP2
 
_user_specified_nameinputs
?

?
&__inference_model_layer_call_fn_435143

inputs
unknown:
?E? 
	unknown_0:?e
	unknown_1:e
	unknown_2:P
	unknown_3:P#
	unknown_4:2
	unknown_5:2#
	unknown_6:2
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????eP*+
_read_only_resource_inputs
		*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4348302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????eP2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_7_layer_call_fn_435206

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????e* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_4345862
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????e2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????e:S O
+
_output_shapes
:?????????e
 
_user_specified_nameinputs
?
?
*__inference_conv1d_15_layer_call_fn_435184

inputs
unknown:?e
	unknown_0:e
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????e*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv1d_15_layer_call_and_return_conditional_losses_4345732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????e2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_434586

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????e2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????e2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????e:S O
+
_output_shapes
:?????????e
 
_user_specified_nameinputs
?
?
*__inference_conv2d_14_layer_call_fn_435256

inputs!
unknown:2
	unknown_0:2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????eP2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_4346232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????eP22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????eP: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????eP
 
_user_specified_nameinputs
?
?
E__inference_conv1d_16_layer_call_and_return_conditional_losses_434604

inputsA
+conv1d_expanddims_1_readvariableop_resource:P-
biasadd_readvariableop_resource:P
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
T0*/
_output_shapes
:?????????e2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:P*
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
:P2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????eP*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????eP*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????eP2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????eP2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????eP2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????e: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????e
 
_user_specified_nameinputs
?

?
&__inference_model_layer_call_fn_435120

inputs
unknown:
?E? 
	unknown_0:?e
	unknown_1:e
	unknown_2:P
	unknown_3:P#
	unknown_4:2
	unknown_5:2#
	unknown_6:2
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????eP*+
_read_only_resource_inputs
		*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4346552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????eP2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_8_layer_call_fn_435278

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????eP2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_4346342
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????eP22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????eP2:W S
/
_output_shapes
:?????????eP2
 
_user_specified_nameinputs
?M
?

"__inference__traced_restore_435443
file_prefix;
'assignvariableop_embedding_8_embeddings:
?E?:
#assignvariableop_1_conv1d_15_kernel:?e/
!assignvariableop_2_conv1d_15_bias:e9
#assignvariableop_3_conv1d_16_kernel:P/
!assignvariableop_4_conv1d_16_bias:P=
#assignvariableop_5_conv2d_14_kernel:2/
!assignvariableop_6_conv2d_14_bias:2=
#assignvariableop_7_conv2d_15_kernel:2/
!assignvariableop_8_conv2d_15_bias:&
assignvariableop_9_adam_iter:	 )
assignvariableop_10_adam_beta_1: )
assignvariableop_11_adam_beta_2: (
assignvariableop_12_adam_decay: 0
&assignvariableop_13_adam_learning_rate: #
assignvariableop_14_total: #
assignvariableop_15_count: %
assignvariableop_16_total_1: %
assignvariableop_17_count_1: 
identity_19??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp'assignvariableop_embedding_8_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp#assignvariableop_1_conv1d_15_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_conv1d_15_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp#assignvariableop_3_conv1d_16_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_conv1d_16_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_conv2d_14_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_conv2d_14_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_conv2d_15_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_conv2d_15_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_iterIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_decayIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp&assignvariableop_13_adam_learning_rateIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_179
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_18f
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_19?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
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
?
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_435261

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????eP22

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????eP22

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????eP2:W S
/
_output_shapes
:?????????eP2
 
_user_specified_nameinputs
?

?
G__inference_embedding_8_layer_call_and_return_conditional_losses_434554

inputs+
embedding_lookup_434548:
?E?
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_434548Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0**
_class 
loc:@embedding_lookup/434548*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@embedding_lookup/434548*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
$__inference_signature_wrapper_434975
input_10
unknown:
?E? 
	unknown_0:?e
	unknown_1:e
	unknown_2:P
	unknown_3:P#
	unknown_4:2
	unknown_5:2#
	unknown_6:2
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????eP*+
_read_only_resource_inputs
		*2
config_proto" 

CPU

GPU2*0,1J 8? **
f%R#
!__inference__wrapped_model_4345372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????eP2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_10
?L
?
A__inference_model_layer_call_and_return_conditional_losses_435029

inputs7
#embedding_8_embedding_lookup_434979:
?E?L
5conv1d_15_conv1d_expanddims_1_readvariableop_resource:?e7
)conv1d_15_biasadd_readvariableop_resource:eK
5conv1d_16_conv1d_expanddims_1_readvariableop_resource:P7
)conv1d_16_biasadd_readvariableop_resource:PB
(conv2d_14_conv2d_readvariableop_resource:27
)conv2d_14_biasadd_readvariableop_resource:2B
(conv2d_15_conv2d_readvariableop_resource:27
)conv2d_15_biasadd_readvariableop_resource:
identity?? conv1d_15/BiasAdd/ReadVariableOp?,conv1d_15/conv1d/ExpandDims_1/ReadVariableOp? conv1d_16/BiasAdd/ReadVariableOp?,conv1d_16/conv1d/ExpandDims_1/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?embedding_8/embedding_lookupu
embedding_8/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_8/Cast?
embedding_8/embedding_lookupResourceGather#embedding_8_embedding_lookup_434979embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_8/embedding_lookup/434979*,
_output_shapes
:??????????*
dtype02
embedding_8/embedding_lookup?
%embedding_8/embedding_lookup/IdentityIdentity%embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_8/embedding_lookup/434979*,
_output_shapes
:??????????2'
%embedding_8/embedding_lookup/Identity?
'embedding_8/embedding_lookup/Identity_1Identity.embedding_8/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2)
'embedding_8/embedding_lookup/Identity_1?
conv1d_15/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_15/conv1d/ExpandDims/dim?
conv1d_15/conv1d/ExpandDims
ExpandDims0embedding_8/embedding_lookup/Identity_1:output:0(conv1d_15/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d_15/conv1d/ExpandDims?
,conv1d_15/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_15_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?e*
dtype02.
,conv1d_15/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_15/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_15/conv1d/ExpandDims_1/dim?
conv1d_15/conv1d/ExpandDims_1
ExpandDims4conv1d_15/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_15/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?e2
conv1d_15/conv1d/ExpandDims_1?
conv1d_15/conv1dConv2D$conv1d_15/conv1d/ExpandDims:output:0&conv1d_15/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????e*
paddingSAME*
strides
2
conv1d_15/conv1d?
conv1d_15/conv1d/SqueezeSqueezeconv1d_15/conv1d:output:0*
T0*+
_output_shapes
:?????????e*
squeeze_dims

?????????2
conv1d_15/conv1d/Squeeze?
 conv1d_15/BiasAdd/ReadVariableOpReadVariableOp)conv1d_15_biasadd_readvariableop_resource*
_output_shapes
:e*
dtype02"
 conv1d_15/BiasAdd/ReadVariableOp?
conv1d_15/BiasAddBiasAdd!conv1d_15/conv1d/Squeeze:output:0(conv1d_15/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????e2
conv1d_15/BiasAdd?
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2)
'tf.compat.v1.transpose_1/transpose/perm?
"tf.compat.v1.transpose_1/transpose	Transposeconv1d_15/BiasAdd:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????e2$
"tf.compat.v1.transpose_1/transpose?
dropout_7/IdentityIdentity&tf.compat.v1.transpose_1/transpose:y:0*
T0*+
_output_shapes
:?????????e2
dropout_7/Identity?
conv1d_16/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_16/conv1d/ExpandDims/dim?
conv1d_16/conv1d/ExpandDims
ExpandDimsdropout_7/Identity:output:0(conv1d_16/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????e2
conv1d_16/conv1d/ExpandDims?
,conv1d_16/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_16_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:P*
dtype02.
,conv1d_16/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_16/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_16/conv1d/ExpandDims_1/dim?
conv1d_16/conv1d/ExpandDims_1
ExpandDims4conv1d_16/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_16/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:P2
conv1d_16/conv1d/ExpandDims_1?
conv1d_16/conv1dConv2D$conv1d_16/conv1d/ExpandDims:output:0&conv1d_16/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????eP*
paddingSAME*
strides
2
conv1d_16/conv1d?
conv1d_16/conv1d/SqueezeSqueezeconv1d_16/conv1d:output:0*
T0*+
_output_shapes
:?????????eP*
squeeze_dims

?????????2
conv1d_16/conv1d/Squeeze?
 conv1d_16/BiasAdd/ReadVariableOpReadVariableOp)conv1d_16_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02"
 conv1d_16/BiasAdd/ReadVariableOp?
conv1d_16/BiasAddBiasAdd!conv1d_16/conv1d/Squeeze:output:0(conv1d_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????eP2
conv1d_16/BiasAddz
conv1d_16/ReluReluconv1d_16/BiasAdd:output:0*
T0*+
_output_shapes
:?????????eP2
conv1d_16/Relu?
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.expand_dims/ExpandDims/dim?
tf.expand_dims/ExpandDims
ExpandDimsconv1d_16/Relu:activations:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????eP2
tf.expand_dims/ExpandDims?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2D"tf.expand_dims/ExpandDims:output:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP2*
paddingSAME*
strides
2
conv2d_14/Conv2D?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP22
conv2d_14/BiasAdd?
conv2d_14/SigmoidSigmoidconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????eP22
conv2d_14/Sigmoid?
dropout_8/IdentityIdentityconv2d_14/Sigmoid:y:0*
T0*/
_output_shapes
:?????????eP22
dropout_8/Identity?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2Ddropout_8/Identity:output:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP*
paddingSAME*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP2
conv2d_15/BiasAdd?
tf.reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????e   P   2
tf.reshape/Reshape/shape?
tf.reshape/ReshapeReshapeconv2d_15/BiasAdd:output:0!tf.reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????eP2
tf.reshape/Reshapez
IdentityIdentitytf.reshape/Reshape:output:0^NoOp*
T0*+
_output_shapes
:?????????eP2

Identity?
NoOpNoOp!^conv1d_15/BiasAdd/ReadVariableOp-^conv1d_15/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_16/BiasAdd/ReadVariableOp-^conv1d_16/conv1d/ExpandDims_1/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp^embedding_8/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 2D
 conv1d_15/BiasAdd/ReadVariableOp conv1d_15/BiasAdd/ReadVariableOp2\
,conv1d_15/conv1d/ExpandDims_1/ReadVariableOp,conv1d_15/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_16/BiasAdd/ReadVariableOp conv1d_16/BiasAdd/ReadVariableOp2\
,conv1d_16/conv1d/ExpandDims_1/ReadVariableOp,conv1d_16/conv1d/ExpandDims_1/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2<
embedding_8/embedding_lookupembedding_8/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
&__inference_model_layer_call_fn_434874
input_10
unknown:
?E? 
	unknown_0:?e
	unknown_1:e
	unknown_2:P
	unknown_3:P#
	unknown_4:2
	unknown_5:2#
	unknown_6:2
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????eP*+
_read_only_resource_inputs
		*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4348302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????eP2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_10
?
?
,__inference_embedding_8_layer_call_fn_435160

inputs
unknown:
?E?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_embedding_8_layer_call_and_return_conditional_losses_4345542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?`
?
A__inference_model_layer_call_and_return_conditional_losses_435097

inputs7
#embedding_8_embedding_lookup_435033:
?E?L
5conv1d_15_conv1d_expanddims_1_readvariableop_resource:?e7
)conv1d_15_biasadd_readvariableop_resource:eK
5conv1d_16_conv1d_expanddims_1_readvariableop_resource:P7
)conv1d_16_biasadd_readvariableop_resource:PB
(conv2d_14_conv2d_readvariableop_resource:27
)conv2d_14_biasadd_readvariableop_resource:2B
(conv2d_15_conv2d_readvariableop_resource:27
)conv2d_15_biasadd_readvariableop_resource:
identity?? conv1d_15/BiasAdd/ReadVariableOp?,conv1d_15/conv1d/ExpandDims_1/ReadVariableOp? conv1d_16/BiasAdd/ReadVariableOp?,conv1d_16/conv1d/ExpandDims_1/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?embedding_8/embedding_lookupu
embedding_8/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_8/Cast?
embedding_8/embedding_lookupResourceGather#embedding_8_embedding_lookup_435033embedding_8/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_8/embedding_lookup/435033*,
_output_shapes
:??????????*
dtype02
embedding_8/embedding_lookup?
%embedding_8/embedding_lookup/IdentityIdentity%embedding_8/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_8/embedding_lookup/435033*,
_output_shapes
:??????????2'
%embedding_8/embedding_lookup/Identity?
'embedding_8/embedding_lookup/Identity_1Identity.embedding_8/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2)
'embedding_8/embedding_lookup/Identity_1?
conv1d_15/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_15/conv1d/ExpandDims/dim?
conv1d_15/conv1d/ExpandDims
ExpandDims0embedding_8/embedding_lookup/Identity_1:output:0(conv1d_15/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d_15/conv1d/ExpandDims?
,conv1d_15/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_15_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?e*
dtype02.
,conv1d_15/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_15/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_15/conv1d/ExpandDims_1/dim?
conv1d_15/conv1d/ExpandDims_1
ExpandDims4conv1d_15/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_15/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?e2
conv1d_15/conv1d/ExpandDims_1?
conv1d_15/conv1dConv2D$conv1d_15/conv1d/ExpandDims:output:0&conv1d_15/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????e*
paddingSAME*
strides
2
conv1d_15/conv1d?
conv1d_15/conv1d/SqueezeSqueezeconv1d_15/conv1d:output:0*
T0*+
_output_shapes
:?????????e*
squeeze_dims

?????????2
conv1d_15/conv1d/Squeeze?
 conv1d_15/BiasAdd/ReadVariableOpReadVariableOp)conv1d_15_biasadd_readvariableop_resource*
_output_shapes
:e*
dtype02"
 conv1d_15/BiasAdd/ReadVariableOp?
conv1d_15/BiasAddBiasAdd!conv1d_15/conv1d/Squeeze:output:0(conv1d_15/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????e2
conv1d_15/BiasAdd?
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2)
'tf.compat.v1.transpose_1/transpose/perm?
"tf.compat.v1.transpose_1/transpose	Transposeconv1d_15/BiasAdd:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????e2$
"tf.compat.v1.transpose_1/transposew
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_7/dropout/Const?
dropout_7/dropout/MulMul&tf.compat.v1.transpose_1/transpose:y:0 dropout_7/dropout/Const:output:0*
T0*+
_output_shapes
:?????????e2
dropout_7/dropout/Mul?
dropout_7/dropout/ShapeShape&tf.compat.v1.transpose_1/transpose:y:0*
T0*
_output_shapes
:2
dropout_7/dropout/Shape?
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????e*
dtype020
.dropout_7/dropout/random_uniform/RandomUniform?
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_7/dropout/GreaterEqual/y?
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????e2 
dropout_7/dropout/GreaterEqual?
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????e2
dropout_7/dropout/Cast?
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????e2
dropout_7/dropout/Mul_1?
conv1d_16/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
conv1d_16/conv1d/ExpandDims/dim?
conv1d_16/conv1d/ExpandDims
ExpandDimsdropout_7/dropout/Mul_1:z:0(conv1d_16/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????e2
conv1d_16/conv1d/ExpandDims?
,conv1d_16/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_16_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:P*
dtype02.
,conv1d_16/conv1d/ExpandDims_1/ReadVariableOp?
!conv1d_16/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_16/conv1d/ExpandDims_1/dim?
conv1d_16/conv1d/ExpandDims_1
ExpandDims4conv1d_16/conv1d/ExpandDims_1/ReadVariableOp:value:0*conv1d_16/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:P2
conv1d_16/conv1d/ExpandDims_1?
conv1d_16/conv1dConv2D$conv1d_16/conv1d/ExpandDims:output:0&conv1d_16/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????eP*
paddingSAME*
strides
2
conv1d_16/conv1d?
conv1d_16/conv1d/SqueezeSqueezeconv1d_16/conv1d:output:0*
T0*+
_output_shapes
:?????????eP*
squeeze_dims

?????????2
conv1d_16/conv1d/Squeeze?
 conv1d_16/BiasAdd/ReadVariableOpReadVariableOp)conv1d_16_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02"
 conv1d_16/BiasAdd/ReadVariableOp?
conv1d_16/BiasAddBiasAdd!conv1d_16/conv1d/Squeeze:output:0(conv1d_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????eP2
conv1d_16/BiasAddz
conv1d_16/ReluReluconv1d_16/BiasAdd:output:0*
T0*+
_output_shapes
:?????????eP2
conv1d_16/Relu?
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.expand_dims/ExpandDims/dim?
tf.expand_dims/ExpandDims
ExpandDimsconv1d_16/Relu:activations:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????eP2
tf.expand_dims/ExpandDims?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2D"tf.expand_dims/ExpandDims:output:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP2*
paddingSAME*
strides
2
conv2d_14/Conv2D?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP22
conv2d_14/BiasAdd?
conv2d_14/SigmoidSigmoidconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????eP22
conv2d_14/Sigmoidw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_8/dropout/Const?
dropout_8/dropout/MulMulconv2d_14/Sigmoid:y:0 dropout_8/dropout/Const:output:0*
T0*/
_output_shapes
:?????????eP22
dropout_8/dropout/Mulw
dropout_8/dropout/ShapeShapeconv2d_14/Sigmoid:y:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape?
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????eP2*
dtype020
.dropout_8/dropout/random_uniform/RandomUniform?
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_8/dropout/GreaterEqual/y?
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????eP22 
dropout_8/dropout/GreaterEqual?
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????eP22
dropout_8/dropout/Cast?
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????eP22
dropout_8/dropout/Mul_1?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2Ddropout_8/dropout/Mul_1:z:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP*
paddingSAME*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP2
conv2d_15/BiasAdd?
tf.reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????e   P   2
tf.reshape/Reshape/shape?
tf.reshape/ReshapeReshapeconv2d_15/BiasAdd:output:0!tf.reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????eP2
tf.reshape/Reshapez
IdentityIdentitytf.reshape/Reshape:output:0^NoOp*
T0*+
_output_shapes
:?????????eP2

Identity?
NoOpNoOp!^conv1d_15/BiasAdd/ReadVariableOp-^conv1d_15/conv1d/ExpandDims_1/ReadVariableOp!^conv1d_16/BiasAdd/ReadVariableOp-^conv1d_16/conv1d/ExpandDims_1/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp^embedding_8/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 2D
 conv1d_15/BiasAdd/ReadVariableOp conv1d_15/BiasAdd/ReadVariableOp2\
,conv1d_15/conv1d/ExpandDims_1/ReadVariableOp,conv1d_15/conv1d/ExpandDims_1/ReadVariableOp2D
 conv1d_16/BiasAdd/ReadVariableOp conv1d_16/BiasAdd/ReadVariableOp2\
,conv1d_16/conv1d/ExpandDims_1/ReadVariableOp,conv1d_16/conv1d/ExpandDims_1/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2<
embedding_8/embedding_lookupembedding_8/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_435189

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????e2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????e2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????e:S O
+
_output_shapes
:?????????e
 
_user_specified_nameinputs
?
?
E__inference_conv1d_15_layer_call_and_return_conditional_losses_434573

inputsB
+conv1d_expanddims_1_readvariableop_resource:?e-
biasadd_readvariableop_resource:e
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
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?e*
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
T0*'
_output_shapes
:?e2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????e*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????e*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:e*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????e2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????e2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_conv1d_16_layer_call_fn_435236

inputs
unknown:P
	unknown_0:P
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????eP*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv1d_16_layer_call_and_return_conditional_losses_4346042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????eP2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????e: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????e
 
_user_specified_nameinputs
?
c
*__inference_dropout_8_layer_call_fn_435283

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????eP2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_4347062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????eP22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????eP222
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????eP2
 
_user_specified_nameinputs
?,
?
A__inference_model_layer_call_and_return_conditional_losses_434655

inputs&
embedding_8_434555:
?E?'
conv1d_15_434574:?e
conv1d_15_434576:e&
conv1d_16_434605:P
conv1d_16_434607:P*
conv2d_14_434624:2
conv2d_14_434626:2*
conv2d_15_434647:2
conv2d_15_434649:
identity??!conv1d_15/StatefulPartitionedCall?!conv1d_16/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?#embedding_8/StatefulPartitionedCall?
#embedding_8/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_8_434555*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_embedding_8_layer_call_and_return_conditional_losses_4345542%
#embedding_8/StatefulPartitionedCall?
!conv1d_15/StatefulPartitionedCallStatefulPartitionedCall,embedding_8/StatefulPartitionedCall:output:0conv1d_15_434574conv1d_15_434576*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????e*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv1d_15_layer_call_and_return_conditional_losses_4345732#
!conv1d_15/StatefulPartitionedCall?
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2)
'tf.compat.v1.transpose_1/transpose/perm?
"tf.compat.v1.transpose_1/transpose	Transpose*conv1d_15/StatefulPartitionedCall:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????e2$
"tf.compat.v1.transpose_1/transpose?
dropout_7/PartitionedCallPartitionedCall&tf.compat.v1.transpose_1/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????e* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_4345862
dropout_7/PartitionedCall?
!conv1d_16/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0conv1d_16_434605conv1d_16_434607*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????eP*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv1d_16_layer_call_and_return_conditional_losses_4346042#
!conv1d_16/StatefulPartitionedCall?
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.expand_dims/ExpandDims/dim?
tf.expand_dims/ExpandDims
ExpandDims*conv1d_16/StatefulPartitionedCall:output:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????eP2
tf.expand_dims/ExpandDims?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall"tf.expand_dims/ExpandDims:output:0conv2d_14_434624conv2d_14_434626*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????eP2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_4346232#
!conv2d_14/StatefulPartitionedCall?
dropout_8/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????eP2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_4346342
dropout_8/PartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0conv2d_15_434647conv2d_15_434649*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????eP*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_4346462#
!conv2d_15/StatefulPartitionedCall?
tf.reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????e   P   2
tf.reshape/Reshape/shape?
tf.reshape/ReshapeReshape*conv2d_15/StatefulPartitionedCall:output:0!tf.reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????eP2
tf.reshape/Reshapez
IdentityIdentitytf.reshape/Reshape:output:0^NoOp*
T0*+
_output_shapes
:?????????eP2

Identity?
NoOpNoOp"^conv1d_15/StatefulPartitionedCall"^conv1d_16/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall$^embedding_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????: : : : : : : : : 2F
!conv1d_15/StatefulPartitionedCall!conv1d_15/StatefulPartitionedCall2F
!conv1d_16/StatefulPartitionedCall!conv1d_16/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2J
#embedding_8/StatefulPartitionedCall#embedding_8/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_8_layer_call_and_return_conditional_losses_435273

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????eP22
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????eP2*
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
T0*/
_output_shapes
:?????????eP22
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????eP22
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????eP22
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????eP22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????eP2:W S
/
_output_shapes
:?????????eP2
 
_user_specified_nameinputs
?
c
*__inference_dropout_7_layer_call_fn_435211

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????e* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_4347492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????e2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????e22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????e
 
_user_specified_nameinputs
?
d
E__inference_dropout_7_layer_call_and_return_conditional_losses_434749

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????e2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????e*
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
T0*+
_output_shapes
:?????????e2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????e2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????e2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????e2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????e:S O
+
_output_shapes
:?????????e
 
_user_specified_nameinputs
?
?
*__inference_conv2d_15_layer_call_fn_435302

inputs!
unknown:2
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????eP*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_4346462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????eP2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????eP2: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????eP2
 
_user_specified_nameinputs
?
?
E__inference_conv2d_15_layer_call_and_return_conditional_losses_435293

inputs8
conv2d_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:2*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????eP2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????eP2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????eP2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????eP2
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
=
input_101
serving_default_input_10:0?????????B

tf.reshape4
StatefulPartitionedCall:0?????????ePtensorflow/serving/predict:ч
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
*r&call_and_return_all_conditional_losses
s__call__
t_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

embeddings
trainable_variables
regularization_losses
	variables
	keras_api
*u&call_and_return_all_conditional_losses
v__call__"
_tf_keras_layer
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*w&call_and_return_all_conditional_losses
x__call__"
_tf_keras_layer
(
	keras_api"
_tf_keras_layer
?
trainable_variables
regularization_losses
 	variables
!	keras_api
*y&call_and_return_all_conditional_losses
z__call__"
_tf_keras_layer
?

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
*{&call_and_return_all_conditional_losses
|__call__"
_tf_keras_layer
(
(	keras_api"
_tf_keras_layer
?

)kernel
*bias
+trainable_variables
,regularization_losses
-	variables
.	keras_api
*}&call_and_return_all_conditional_losses
~__call__"
_tf_keras_layer
?
/trainable_variables
0regularization_losses
1	variables
2	keras_api
*&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
(
9	keras_api"
_tf_keras_layer
S
:iter

;beta_1

<beta_2
	=decay
>learning_rate"
	optimizer
_
0
1
2
"3
#4
)5
*6
37
48"
trackable_list_wrapper
 "
trackable_list_wrapper
_
0
1
2
"3
#4
)5
*6
37
48"
trackable_list_wrapper
?

?layers
trainable_variables
@layer_metrics
Anon_trainable_variables
regularization_losses
Bmetrics
	variables
Clayer_regularization_losses
s__call__
t_default_save_signature
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:(
?E?2embedding_8/embeddings
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?

Dlayers
trainable_variables
Elayer_metrics
Fnon_trainable_variables
regularization_losses
Gmetrics
	variables
Hlayer_regularization_losses
v__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
':%?e2conv1d_15/kernel
:e2conv1d_15/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Ilayers
trainable_variables
Jlayer_metrics
Knon_trainable_variables
regularization_losses
Lmetrics
	variables
Mlayer_regularization_losses
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Nlayers
trainable_variables
Olayer_metrics
Pnon_trainable_variables
regularization_losses
Qmetrics
 	variables
Rlayer_regularization_losses
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
&:$P2conv1d_16/kernel
:P2conv1d_16/bias
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?

Slayers
$trainable_variables
Tlayer_metrics
Unon_trainable_variables
%regularization_losses
Vmetrics
&	variables
Wlayer_regularization_losses
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
*:(22conv2d_14/kernel
:22conv2d_14/bias
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?

Xlayers
+trainable_variables
Ylayer_metrics
Znon_trainable_variables
,regularization_losses
[metrics
-	variables
\layer_regularization_losses
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

]layers
/trainable_variables
^layer_metrics
_non_trainable_variables
0regularization_losses
`metrics
1	variables
alayer_regularization_losses
?__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
*:(22conv2d_15/kernel
:2conv2d_15/bias
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?

blayers
5trainable_variables
clayer_metrics
dnon_trainable_variables
6regularization_losses
emetrics
7	variables
flayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
n
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
10"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
g0
h1"
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
N
	itotal
	jcount
k	variables
l	keras_api"
_tf_keras_metric
^
	mtotal
	ncount
o
_fn_kwargs
p	variables
q	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
i0
j1"
trackable_list_wrapper
-
k	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
m0
n1"
trackable_list_wrapper
-
p	variables"
_generic_user_object
?2?
A__inference_model_layer_call_and_return_conditional_losses_435029
A__inference_model_layer_call_and_return_conditional_losses_435097
A__inference_model_layer_call_and_return_conditional_losses_434909
A__inference_model_layer_call_and_return_conditional_losses_434944?
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
&__inference_model_layer_call_fn_434676
&__inference_model_layer_call_fn_435120
&__inference_model_layer_call_fn_435143
&__inference_model_layer_call_fn_434874?
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
?B?
!__inference__wrapped_model_434537input_10"?
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
?2?
G__inference_embedding_8_layer_call_and_return_conditional_losses_435153?
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
,__inference_embedding_8_layer_call_fn_435160?
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
E__inference_conv1d_15_layer_call_and_return_conditional_losses_435175?
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
*__inference_conv1d_15_layer_call_fn_435184?
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
E__inference_dropout_7_layer_call_and_return_conditional_losses_435189
E__inference_dropout_7_layer_call_and_return_conditional_losses_435201?
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
*__inference_dropout_7_layer_call_fn_435206
*__inference_dropout_7_layer_call_fn_435211?
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
E__inference_conv1d_16_layer_call_and_return_conditional_losses_435227?
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
*__inference_conv1d_16_layer_call_fn_435236?
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
E__inference_conv2d_14_layer_call_and_return_conditional_losses_435247?
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
*__inference_conv2d_14_layer_call_fn_435256?
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
E__inference_dropout_8_layer_call_and_return_conditional_losses_435261
E__inference_dropout_8_layer_call_and_return_conditional_losses_435273?
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
*__inference_dropout_8_layer_call_fn_435278
*__inference_dropout_8_layer_call_fn_435283?
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
E__inference_conv2d_15_layer_call_and_return_conditional_losses_435293?
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
*__inference_conv2d_15_layer_call_fn_435302?
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
$__inference_signature_wrapper_434975input_10"?
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
!__inference__wrapped_model_434537{	"#)*341?.
'?$
"?
input_10?????????
? ";?8
6

tf.reshape(?%

tf.reshape?????????eP?
E__inference_conv1d_15_layer_call_and_return_conditional_losses_435175e4?1
*?'
%?"
inputs??????????
? ")?&
?
0?????????e
? ?
*__inference_conv1d_15_layer_call_fn_435184X4?1
*?'
%?"
inputs??????????
? "??????????e?
E__inference_conv1d_16_layer_call_and_return_conditional_losses_435227d"#3?0
)?&
$?!
inputs?????????e
? ")?&
?
0?????????eP
? ?
*__inference_conv1d_16_layer_call_fn_435236W"#3?0
)?&
$?!
inputs?????????e
? "??????????eP?
E__inference_conv2d_14_layer_call_and_return_conditional_losses_435247l)*7?4
-?*
(?%
inputs?????????eP
? "-?*
#? 
0?????????eP2
? ?
*__inference_conv2d_14_layer_call_fn_435256_)*7?4
-?*
(?%
inputs?????????eP
? " ??????????eP2?
E__inference_conv2d_15_layer_call_and_return_conditional_losses_435293l347?4
-?*
(?%
inputs?????????eP2
? "-?*
#? 
0?????????eP
? ?
*__inference_conv2d_15_layer_call_fn_435302_347?4
-?*
(?%
inputs?????????eP2
? " ??????????eP?
E__inference_dropout_7_layer_call_and_return_conditional_losses_435189d7?4
-?*
$?!
inputs?????????e
p 
? ")?&
?
0?????????e
? ?
E__inference_dropout_7_layer_call_and_return_conditional_losses_435201d7?4
-?*
$?!
inputs?????????e
p
? ")?&
?
0?????????e
? ?
*__inference_dropout_7_layer_call_fn_435206W7?4
-?*
$?!
inputs?????????e
p 
? "??????????e?
*__inference_dropout_7_layer_call_fn_435211W7?4
-?*
$?!
inputs?????????e
p
? "??????????e?
E__inference_dropout_8_layer_call_and_return_conditional_losses_435261l;?8
1?.
(?%
inputs?????????eP2
p 
? "-?*
#? 
0?????????eP2
? ?
E__inference_dropout_8_layer_call_and_return_conditional_losses_435273l;?8
1?.
(?%
inputs?????????eP2
p
? "-?*
#? 
0?????????eP2
? ?
*__inference_dropout_8_layer_call_fn_435278_;?8
1?.
(?%
inputs?????????eP2
p 
? " ??????????eP2?
*__inference_dropout_8_layer_call_fn_435283_;?8
1?.
(?%
inputs?????????eP2
p
? " ??????????eP2?
G__inference_embedding_8_layer_call_and_return_conditional_losses_435153`/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
,__inference_embedding_8_layer_call_fn_435160S/?,
%?"
 ?
inputs?????????
? "????????????
A__inference_model_layer_call_and_return_conditional_losses_434909q	"#)*349?6
/?,
"?
input_10?????????
p 

 
? ")?&
?
0?????????eP
? ?
A__inference_model_layer_call_and_return_conditional_losses_434944q	"#)*349?6
/?,
"?
input_10?????????
p

 
? ")?&
?
0?????????eP
? ?
A__inference_model_layer_call_and_return_conditional_losses_435029o	"#)*347?4
-?*
 ?
inputs?????????
p 

 
? ")?&
?
0?????????eP
? ?
A__inference_model_layer_call_and_return_conditional_losses_435097o	"#)*347?4
-?*
 ?
inputs?????????
p

 
? ")?&
?
0?????????eP
? ?
&__inference_model_layer_call_fn_434676d	"#)*349?6
/?,
"?
input_10?????????
p 

 
? "??????????eP?
&__inference_model_layer_call_fn_434874d	"#)*349?6
/?,
"?
input_10?????????
p

 
? "??????????eP?
&__inference_model_layer_call_fn_435120b	"#)*347?4
-?*
 ?
inputs?????????
p 

 
? "??????????eP?
&__inference_model_layer_call_fn_435143b	"#)*347?4
-?*
 ?
inputs?????????
p

 
? "??????????eP?
$__inference_signature_wrapper_434975?	"#)*34=?:
? 
3?0
.
input_10"?
input_10?????????";?8
6

tf.reshape(?%

tf.reshape?????????eP