�
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
-
Sqrt
x"T
y"T"
Ttype:

2
�
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
executor_typestring ��
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��

�
ConstConst*
_output_shapes

:*
dtype0*Y
valuePBN"@N5�Q��@���B41�A�$�)h�#FlA�&3B��Q;�@sNB�R�A�?X7��AF�2Ae�B
�
Const_1Const*
_output_shapes

:*
dtype0*Y
valuePBN"@=I�#�@�X+A���@?�4z��@L�>A��A��IW�@�l�@/�,@׵�7�I@S�LA��A
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
�
Adam/v/dense_83/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_83/bias
y
(Adam/v/dense_83/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_83/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_83/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_83/bias
y
(Adam/m/dense_83/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_83/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_83/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/v/dense_83/kernel
�
*Adam/v/dense_83/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_83/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_83/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/m/dense_83/kernel
�
*Adam/m/dense_83/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_83/kernel*
_output_shapes
:	�*
dtype0
�
Adam/v/dense_82/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/dense_82/bias
z
(Adam/v/dense_82/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_82/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_82/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/dense_82/bias
z
(Adam/m/dense_82/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_82/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_82/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/v/dense_82/kernel
�
*Adam/v/dense_82/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_82/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_82/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/m/dense_82/kernel
�
*Adam/m/dense_82/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_82/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_81/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/dense_81/bias
z
(Adam/v/dense_81/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_81/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_81/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/dense_81/bias
z
(Adam/m/dense_81/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_81/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_81/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/v/dense_81/kernel
�
*Adam/v/dense_81/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_81/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_81/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/m/dense_81/kernel
�
*Adam/m/dense_81/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_81/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_80/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/dense_80/bias
z
(Adam/v/dense_80/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_80/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_80/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/dense_80/bias
z
(Adam/m/dense_80/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_80/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_80/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/v/dense_80/kernel
�
*Adam/v/dense_80/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_80/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_80/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/m/dense_80/kernel
�
*Adam/m/dense_80/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_80/kernel*
_output_shapes
:	�*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
r
dense_83/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_83/bias
k
!dense_83/bias/Read/ReadVariableOpReadVariableOpdense_83/bias*
_output_shapes
:*
dtype0
{
dense_83/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_83/kernel
t
#dense_83/kernel/Read/ReadVariableOpReadVariableOpdense_83/kernel*
_output_shapes
:	�*
dtype0
s
dense_82/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_82/bias
l
!dense_82/bias/Read/ReadVariableOpReadVariableOpdense_82/bias*
_output_shapes	
:�*
dtype0
|
dense_82/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_82/kernel
u
#dense_82/kernel/Read/ReadVariableOpReadVariableOpdense_82/kernel* 
_output_shapes
:
��*
dtype0
s
dense_81/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_81/bias
l
!dense_81/bias/Read/ReadVariableOpReadVariableOpdense_81/bias*
_output_shapes	
:�*
dtype0
|
dense_81/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_81/kernel
u
#dense_81/kernel/Read/ReadVariableOpReadVariableOpdense_81/kernel* 
_output_shapes
:
��*
dtype0
s
dense_80/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_80/bias
l
!dense_80/bias/Read/ReadVariableOpReadVariableOpdense_80/bias*
_output_shapes	
:�*
dtype0
{
dense_80/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_80/kernel
t
#dense_80/kernel/Read/ReadVariableOpReadVariableOpdense_80/kernel*
_output_shapes
:	�*
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0	
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
dtype0
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
�
#serving_default_normalization_inputPlaceholder*0
_output_shapes
:������������������*
dtype0*%
shape:������������������
�
StatefulPartitionedCallStatefulPartitionedCall#serving_default_normalization_inputConst_1Constdense_80/kerneldense_80/biasdense_81/kerneldense_81/biasdense_82/kerneldense_82/biasdense_83/kerneldense_83/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_56761540

NoOpNoOp
�H
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*�H
value�HB�H B�H
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
_adapt_function*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias*
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)_random_generator* 
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

0kernel
1bias*
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
8_random_generator* 
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias*
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
G_random_generator* 
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias*
R
0
1
2
!3
"4
05
16
?7
@8
N9
O10*
<
!0
"1
02
13
?4
@5
N6
O7*

P0
Q1
R2* 
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Xtrace_0
Ytrace_1
Ztrace_2
[trace_3* 
6
\trace_0
]trace_1
^trace_2
_trace_3* 
 
`	capture_0
a	capture_1* 
�
b
_variables
c_iterations
d_learning_rate
e_index_dict
f
_momentums
g_velocities
h_update_step_xla*

iserving_default* 
* 
* 
* 
* 
* 
RL
VARIABLE_VALUEmean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEvariance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEcount_15layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE*

jtrace_0* 

!0
"1*

!0
"1*
	
P0* 
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*

ptrace_0* 

qtrace_0* 
_Y
VARIABLE_VALUEdense_80/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_80/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses* 

wtrace_0
xtrace_1* 

ytrace_0
ztrace_1* 
* 

00
11*

00
11*
	
Q0* 
�
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_81/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_81/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

?0
@1*

?0
@1*
	
R0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_82/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_82/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

N0
O1*

N0
O1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_83/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_83/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 

�trace_0* 

0
1
2*
<
0
1
2
3
4
5
6
7*

�0*
* 
* 
 
`	capture_0
a	capture_1* 
 
`	capture_0
a	capture_1* 
 
`	capture_0
a	capture_1* 
 
`	capture_0
a	capture_1* 
 
`	capture_0
a	capture_1* 
 
`	capture_0
a	capture_1* 
 
`	capture_0
a	capture_1* 
 
`	capture_0
a	capture_1* 
* 
* 
�
c0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
D
�0
�1
�2
�3
�4
�5
�6
�7*
D
�0
�1
�2
�3
�4
�5
�6
�7*
* 
 
`	capture_0
a	capture_1* 
* 
* 
* 
* 
	
P0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
Q0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
R0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
a[
VARIABLE_VALUEAdam/m/dense_80/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_80/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_80/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_80/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_81/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_81/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_81/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_81/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_82/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_82/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_82/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_82/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_83/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_83/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_83/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_83/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemeanvariancecount_1dense_80/kerneldense_80/biasdense_81/kerneldense_81/biasdense_82/kerneldense_82/biasdense_83/kerneldense_83/bias	iterationlearning_rateAdam/m/dense_80/kernelAdam/v/dense_80/kernelAdam/m/dense_80/biasAdam/v/dense_80/biasAdam/m/dense_81/kernelAdam/v/dense_81/kernelAdam/m/dense_81/biasAdam/v/dense_81/biasAdam/m/dense_82/kernelAdam/v/dense_82/kernelAdam/m/dense_82/biasAdam/v/dense_82/biasAdam/m/dense_83/kernelAdam/v/dense_83/kernelAdam/m/dense_83/biasAdam/v/dense_83/biastotalcountConst_2*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_save_56762139
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecount_1dense_80/kerneldense_80/biasdense_81/kerneldense_81/biasdense_82/kerneldense_82/biasdense_83/kerneldense_83/bias	iterationlearning_rateAdam/m/dense_80/kernelAdam/v/dense_80/kernelAdam/m/dense_80/biasAdam/v/dense_80/biasAdam/m/dense_81/kernelAdam/v/dense_81/kernelAdam/m/dense_81/biasAdam/v/dense_81/biasAdam/m/dense_82/kernelAdam/v/dense_82/kernelAdam/m/dense_82/biasAdam/v/dense_82/biasAdam/m/dense_83/kernelAdam/v/dense_83/kernelAdam/m/dense_83/biasAdam/v/dense_83/biastotalcount*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__traced_restore_56762242��	
�

�
0__inference_sequential_20_layer_call_fn_56761602

inputs
unknown
	unknown_0
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761353o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:������������������::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_output_shapes

::$ 

_output_shapes

::X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�V
�
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761676

inputs
normalization_sub_y
normalization_sqrt_x:
'dense_80_matmul_readvariableop_resource:	�7
(dense_80_biasadd_readvariableop_resource:	�;
'dense_81_matmul_readvariableop_resource:
��7
(dense_81_biasadd_readvariableop_resource:	�;
'dense_82_matmul_readvariableop_resource:
��7
(dense_82_biasadd_readvariableop_resource:	�:
'dense_83_matmul_readvariableop_resource:	�6
(dense_83_biasadd_readvariableop_resource:
identity��dense_80/BiasAdd/ReadVariableOp�dense_80/MatMul/ReadVariableOp�1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp�dense_81/BiasAdd/ReadVariableOp�dense_81/MatMul/ReadVariableOp�1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp�dense_82/BiasAdd/ReadVariableOp�dense_82/MatMul/ReadVariableOp�1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp�dense_83/BiasAdd/ReadVariableOp�dense_83/MatMul/ReadVariableOpg
normalization/subSubinputsnormalization_sub_y*
T0*'
_output_shapes
:���������Y
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:�
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:����������
dense_80/MatMul/ReadVariableOpReadVariableOp'dense_80_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_80/MatMulMatMulnormalization/truediv:z:0&dense_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_80/BiasAdd/ReadVariableOpReadVariableOp(dense_80_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_80/BiasAddBiasAdddense_80/MatMul:product:0'dense_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_80/ReluReludense_80/BiasAdd:output:0*
T0*(
_output_shapes
:����������]
dropout_60/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
dropout_60/dropout/MulMuldense_80/Relu:activations:0!dropout_60/dropout/Const:output:0*
T0*(
_output_shapes
:����������q
dropout_60/dropout/ShapeShapedense_80/Relu:activations:0*
T0*
_output_shapes
::���
/dropout_60/dropout/random_uniform/RandomUniformRandomUniform!dropout_60/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0f
!dropout_60/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout_60/dropout/GreaterEqualGreaterEqual8dropout_60/dropout/random_uniform/RandomUniform:output:0*dropout_60/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������_
dropout_60/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_60/dropout/SelectV2SelectV2#dropout_60/dropout/GreaterEqual:z:0dropout_60/dropout/Mul:z:0#dropout_60/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
dense_81/MatMul/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_81/MatMulMatMul$dropout_60/dropout/SelectV2:output:0&dense_81/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_81/BiasAdd/ReadVariableOpReadVariableOp(dense_81_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_81/BiasAddBiasAdddense_81/MatMul:product:0'dense_81/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_81/ReluReludense_81/BiasAdd:output:0*
T0*(
_output_shapes
:����������]
dropout_61/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
dropout_61/dropout/MulMuldense_81/Relu:activations:0!dropout_61/dropout/Const:output:0*
T0*(
_output_shapes
:����������q
dropout_61/dropout/ShapeShapedense_81/Relu:activations:0*
T0*
_output_shapes
::���
/dropout_61/dropout/random_uniform/RandomUniformRandomUniform!dropout_61/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0f
!dropout_61/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout_61/dropout/GreaterEqualGreaterEqual8dropout_61/dropout/random_uniform/RandomUniform:output:0*dropout_61/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������_
dropout_61/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_61/dropout/SelectV2SelectV2#dropout_61/dropout/GreaterEqual:z:0dropout_61/dropout/Mul:z:0#dropout_61/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
dense_82/MatMul/ReadVariableOpReadVariableOp'dense_82_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_82/MatMulMatMul$dropout_61/dropout/SelectV2:output:0&dense_82/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_82/BiasAdd/ReadVariableOpReadVariableOp(dense_82_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_82/BiasAddBiasAdddense_82/MatMul:product:0'dense_82/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_82/ReluReludense_82/BiasAdd:output:0*
T0*(
_output_shapes
:����������]
dropout_62/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
dropout_62/dropout/MulMuldense_82/Relu:activations:0!dropout_62/dropout/Const:output:0*
T0*(
_output_shapes
:����������q
dropout_62/dropout/ShapeShapedense_82/Relu:activations:0*
T0*
_output_shapes
::���
/dropout_62/dropout/random_uniform/RandomUniformRandomUniform!dropout_62/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0f
!dropout_62/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout_62/dropout/GreaterEqualGreaterEqual8dropout_62/dropout/random_uniform/RandomUniform:output:0*dropout_62/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������_
dropout_62/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_62/dropout/SelectV2SelectV2#dropout_62/dropout/GreaterEqual:z:0dropout_62/dropout/Mul:z:0#dropout_62/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
dense_83/MatMul/ReadVariableOpReadVariableOp'dense_83_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_83/MatMulMatMul$dropout_62/dropout/SelectV2:output:0&dense_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_83/BiasAdd/ReadVariableOpReadVariableOp(dense_83_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_83/BiasAddBiasAdddense_83/MatMul:product:0'dense_83/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1dense_80/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_80_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"dense_80/kernel/Regularizer/L2LossL2Loss9dense_80/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0+dense_80/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_81/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_81/kernel/Regularizer/L2LossL2Loss9dense_81/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0+dense_81/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_82/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_82_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_82/kernel/Regularizer/L2LossL2Loss9dense_82/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_82/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_82/kernel/Regularizer/mulMul*dense_82/kernel/Regularizer/mul/x:output:0+dense_82/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_83/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_80/BiasAdd/ReadVariableOp^dense_80/MatMul/ReadVariableOp2^dense_80/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_81/BiasAdd/ReadVariableOp^dense_81/MatMul/ReadVariableOp2^dense_81/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_82/BiasAdd/ReadVariableOp^dense_82/MatMul/ReadVariableOp2^dense_82/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_83/BiasAdd/ReadVariableOp^dense_83/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:������������������::: : : : : : : : 2B
dense_80/BiasAdd/ReadVariableOpdense_80/BiasAdd/ReadVariableOp2@
dense_80/MatMul/ReadVariableOpdense_80/MatMul/ReadVariableOp2f
1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_81/BiasAdd/ReadVariableOpdense_81/BiasAdd/ReadVariableOp2@
dense_81/MatMul/ReadVariableOpdense_81/MatMul/ReadVariableOp2f
1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_82/BiasAdd/ReadVariableOpdense_82/BiasAdd/ReadVariableOp2@
dense_82/MatMul/ReadVariableOpdense_82/MatMul/ReadVariableOp2f
1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_83/BiasAdd/ReadVariableOpdense_83/BiasAdd/ReadVariableOp2@
dense_83/MatMul/ReadVariableOpdense_83/MatMul/ReadVariableOp:$ 

_output_shapes

::$ 

_output_shapes

::X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
��
�
!__inference__traced_save_56762139
file_prefix)
read_disablecopyonread_mean:/
!read_1_disablecopyonread_variance:*
 read_2_disablecopyonread_count_1:	 ;
(read_3_disablecopyonread_dense_80_kernel:	�5
&read_4_disablecopyonread_dense_80_bias:	�<
(read_5_disablecopyonread_dense_81_kernel:
��5
&read_6_disablecopyonread_dense_81_bias:	�<
(read_7_disablecopyonread_dense_82_kernel:
��5
&read_8_disablecopyonread_dense_82_bias:	�;
(read_9_disablecopyonread_dense_83_kernel:	�5
'read_10_disablecopyonread_dense_83_bias:-
#read_11_disablecopyonread_iteration:	 1
'read_12_disablecopyonread_learning_rate: C
0read_13_disablecopyonread_adam_m_dense_80_kernel:	�C
0read_14_disablecopyonread_adam_v_dense_80_kernel:	�=
.read_15_disablecopyonread_adam_m_dense_80_bias:	�=
.read_16_disablecopyonread_adam_v_dense_80_bias:	�D
0read_17_disablecopyonread_adam_m_dense_81_kernel:
��D
0read_18_disablecopyonread_adam_v_dense_81_kernel:
��=
.read_19_disablecopyonread_adam_m_dense_81_bias:	�=
.read_20_disablecopyonread_adam_v_dense_81_bias:	�D
0read_21_disablecopyonread_adam_m_dense_82_kernel:
��D
0read_22_disablecopyonread_adam_v_dense_82_kernel:
��=
.read_23_disablecopyonread_adam_m_dense_82_bias:	�=
.read_24_disablecopyonread_adam_v_dense_82_bias:	�C
0read_25_disablecopyonread_adam_m_dense_83_kernel:	�C
0read_26_disablecopyonread_adam_v_dense_83_kernel:	�<
.read_27_disablecopyonread_adam_m_dense_83_bias:<
.read_28_disablecopyonread_adam_v_dense_83_bias:)
read_29_disablecopyonread_total: )
read_30_disablecopyonread_count: 
savev2_const_2
identity_63��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: m
Read/DisableCopyOnReadDisableCopyOnReadread_disablecopyonread_mean"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOpread_disablecopyonread_mean^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0e
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:]

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:u
Read_1/DisableCopyOnReadDisableCopyOnRead!read_1_disablecopyonread_variance"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp!read_1_disablecopyonread_variance^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_2/DisableCopyOnReadDisableCopyOnRead read_2_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp read_2_disablecopyonread_count_1^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	e

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: [

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_3/DisableCopyOnReadDisableCopyOnRead(read_3_disablecopyonread_dense_80_kernel"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp(read_3_disablecopyonread_dense_80_kernel^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0n

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�d

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:	�z
Read_4/DisableCopyOnReadDisableCopyOnRead&read_4_disablecopyonread_dense_80_bias"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp&read_4_disablecopyonread_dense_80_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_5/DisableCopyOnReadDisableCopyOnRead(read_5_disablecopyonread_dense_81_kernel"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp(read_5_disablecopyonread_dense_81_kernel^Read_5/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��z
Read_6/DisableCopyOnReadDisableCopyOnRead&read_6_disablecopyonread_dense_81_bias"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp&read_6_disablecopyonread_dense_81_bias^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_7/DisableCopyOnReadDisableCopyOnRead(read_7_disablecopyonread_dense_82_kernel"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp(read_7_disablecopyonread_dense_82_kernel^Read_7/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��z
Read_8/DisableCopyOnReadDisableCopyOnRead&read_8_disablecopyonread_dense_82_bias"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp&read_8_disablecopyonread_dense_82_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_9/DisableCopyOnReadDisableCopyOnRead(read_9_disablecopyonread_dense_83_kernel"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp(read_9_disablecopyonread_dense_83_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0o
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:	�|
Read_10/DisableCopyOnReadDisableCopyOnRead'read_10_disablecopyonread_dense_83_bias"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp'read_10_disablecopyonread_dense_83_bias^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_11/DisableCopyOnReadDisableCopyOnRead#read_11_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp#read_11_disablecopyonread_iteration^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_12/DisableCopyOnReadDisableCopyOnRead'read_12_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp'read_12_disablecopyonread_learning_rate^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_13/DisableCopyOnReadDisableCopyOnRead0read_13_disablecopyonread_adam_m_dense_80_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp0read_13_disablecopyonread_adam_m_dense_80_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_14/DisableCopyOnReadDisableCopyOnRead0read_14_disablecopyonread_adam_v_dense_80_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp0read_14_disablecopyonread_adam_v_dense_80_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_15/DisableCopyOnReadDisableCopyOnRead.read_15_disablecopyonread_adam_m_dense_80_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp.read_15_disablecopyonread_adam_m_dense_80_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_16/DisableCopyOnReadDisableCopyOnRead.read_16_disablecopyonread_adam_v_dense_80_bias"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp.read_16_disablecopyonread_adam_v_dense_80_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_17/DisableCopyOnReadDisableCopyOnRead0read_17_disablecopyonread_adam_m_dense_81_kernel"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp0read_17_disablecopyonread_adam_m_dense_81_kernel^Read_17/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_18/DisableCopyOnReadDisableCopyOnRead0read_18_disablecopyonread_adam_v_dense_81_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp0read_18_disablecopyonread_adam_v_dense_81_kernel^Read_18/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_19/DisableCopyOnReadDisableCopyOnRead.read_19_disablecopyonread_adam_m_dense_81_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp.read_19_disablecopyonread_adam_m_dense_81_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_20/DisableCopyOnReadDisableCopyOnRead.read_20_disablecopyonread_adam_v_dense_81_bias"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp.read_20_disablecopyonread_adam_v_dense_81_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_21/DisableCopyOnReadDisableCopyOnRead0read_21_disablecopyonread_adam_m_dense_82_kernel"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp0read_21_disablecopyonread_adam_m_dense_82_kernel^Read_21/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_22/DisableCopyOnReadDisableCopyOnRead0read_22_disablecopyonread_adam_v_dense_82_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp0read_22_disablecopyonread_adam_v_dense_82_kernel^Read_22/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_23/DisableCopyOnReadDisableCopyOnRead.read_23_disablecopyonread_adam_m_dense_82_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp.read_23_disablecopyonread_adam_m_dense_82_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_24/DisableCopyOnReadDisableCopyOnRead.read_24_disablecopyonread_adam_v_dense_82_bias"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp.read_24_disablecopyonread_adam_v_dense_82_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_25/DisableCopyOnReadDisableCopyOnRead0read_25_disablecopyonread_adam_m_dense_83_kernel"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp0read_25_disablecopyonread_adam_m_dense_83_kernel^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_26/DisableCopyOnReadDisableCopyOnRead0read_26_disablecopyonread_adam_v_dense_83_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp0read_26_disablecopyonread_adam_v_dense_83_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_27/DisableCopyOnReadDisableCopyOnRead.read_27_disablecopyonread_adam_m_dense_83_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp.read_27_disablecopyonread_adam_m_dense_83_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_28/DisableCopyOnReadDisableCopyOnRead.read_28_disablecopyonread_adam_v_dense_83_bias"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp.read_28_disablecopyonread_adam_v_dense_83_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_29/DisableCopyOnReadDisableCopyOnReadread_29_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOpread_29_disablecopyonread_total^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_30/DisableCopyOnReadDisableCopyOnReadread_30_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOpread_30_disablecopyonread_count^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0savev2_const_2"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *.
dtypes$
"2 		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_62Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_63IdentityIdentity_62:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_63Identity_63:output:0*U
_input_shapesD
B: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp: 

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�;
�
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761172
normalization_input
normalization_sub_y
normalization_sqrt_x$
dense_80_56761054:	� 
dense_80_56761056:	�%
dense_81_56761089:
�� 
dense_81_56761091:	�%
dense_82_56761124:
�� 
dense_82_56761126:	�$
dense_83_56761154:	�
dense_83_56761156:
identity�� dense_80/StatefulPartitionedCall�1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp� dense_81/StatefulPartitionedCall�1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp� dense_82/StatefulPartitionedCall�1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp� dense_83/StatefulPartitionedCall�"dropout_60/StatefulPartitionedCall�"dropout_61/StatefulPartitionedCall�"dropout_62/StatefulPartitionedCallt
normalization/subSubnormalization_inputnormalization_sub_y*
T0*'
_output_shapes
:���������Y
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:�
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:����������
 dense_80/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_80_56761054dense_80_56761056*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_80_layer_call_and_return_conditional_losses_56761053�
"dropout_60/StatefulPartitionedCallStatefulPartitionedCall)dense_80/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_60_layer_call_and_return_conditional_losses_56761071�
 dense_81/StatefulPartitionedCallStatefulPartitionedCall+dropout_60/StatefulPartitionedCall:output:0dense_81_56761089dense_81_56761091*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_81_layer_call_and_return_conditional_losses_56761088�
"dropout_61/StatefulPartitionedCallStatefulPartitionedCall)dense_81/StatefulPartitionedCall:output:0#^dropout_60/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_61_layer_call_and_return_conditional_losses_56761106�
 dense_82/StatefulPartitionedCallStatefulPartitionedCall+dropout_61/StatefulPartitionedCall:output:0dense_82_56761124dense_82_56761126*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_82_layer_call_and_return_conditional_losses_56761123�
"dropout_62/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0#^dropout_61/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_62_layer_call_and_return_conditional_losses_56761141�
 dense_83/StatefulPartitionedCallStatefulPartitionedCall+dropout_62/StatefulPartitionedCall:output:0dense_83_56761154dense_83_56761156*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_83_layer_call_and_return_conditional_losses_56761153�
1dense_80/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_80_56761054*
_output_shapes
:	�*
dtype0�
"dense_80/kernel/Regularizer/L2LossL2Loss9dense_80/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0+dense_80/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_81/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_81_56761089* 
_output_shapes
:
��*
dtype0�
"dense_81/kernel/Regularizer/L2LossL2Loss9dense_81/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0+dense_81/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_82/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_82_56761124* 
_output_shapes
:
��*
dtype0�
"dense_82/kernel/Regularizer/L2LossL2Loss9dense_82/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_82/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_82/kernel/Regularizer/mulMul*dense_82/kernel/Regularizer/mul/x:output:0+dense_82/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_83/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_80/StatefulPartitionedCall2^dense_80/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_81/StatefulPartitionedCall2^dense_81/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_82/StatefulPartitionedCall2^dense_82/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_83/StatefulPartitionedCall#^dropout_60/StatefulPartitionedCall#^dropout_61/StatefulPartitionedCall#^dropout_62/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:������������������::: : : : : : : : 2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2f
1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2f
1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2f
1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2H
"dropout_60/StatefulPartitionedCall"dropout_60/StatefulPartitionedCall2H
"dropout_61/StatefulPartitionedCall"dropout_61/StatefulPartitionedCall2H
"dropout_62/StatefulPartitionedCall"dropout_62/StatefulPartitionedCall:$ 

_output_shapes

::$ 

_output_shapes

::e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�

g
H__inference_dropout_61_layer_call_and_return_conditional_losses_56761826

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
F__inference_dense_83_layer_call_and_return_conditional_losses_56761153

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_dense_81_layer_call_fn_56761789

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_81_layer_call_and_return_conditional_losses_56761088p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

g
H__inference_dropout_62_layer_call_and_return_conditional_losses_56761877

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
-__inference_dropout_60_layer_call_fn_56761758

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_60_layer_call_and_return_conditional_losses_56761071p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
I
-__inference_dropout_61_layer_call_fn_56761814

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_61_layer_call_and_return_conditional_losses_56761202a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_dense_81_layer_call_and_return_conditional_losses_56761804

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_81/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
1dense_81/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_81/kernel/Regularizer/L2LossL2Loss9dense_81/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0+dense_81/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_81/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

g
H__inference_dropout_61_layer_call_and_return_conditional_losses_56761106

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

g
H__inference_dropout_60_layer_call_and_return_conditional_losses_56761775

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�8
�
#__inference__wrapped_model_56761027
normalization_input%
!sequential_20_normalization_sub_y&
"sequential_20_normalization_sqrt_xH
5sequential_20_dense_80_matmul_readvariableop_resource:	�E
6sequential_20_dense_80_biasadd_readvariableop_resource:	�I
5sequential_20_dense_81_matmul_readvariableop_resource:
��E
6sequential_20_dense_81_biasadd_readvariableop_resource:	�I
5sequential_20_dense_82_matmul_readvariableop_resource:
��E
6sequential_20_dense_82_biasadd_readvariableop_resource:	�H
5sequential_20_dense_83_matmul_readvariableop_resource:	�D
6sequential_20_dense_83_biasadd_readvariableop_resource:
identity��-sequential_20/dense_80/BiasAdd/ReadVariableOp�,sequential_20/dense_80/MatMul/ReadVariableOp�-sequential_20/dense_81/BiasAdd/ReadVariableOp�,sequential_20/dense_81/MatMul/ReadVariableOp�-sequential_20/dense_82/BiasAdd/ReadVariableOp�,sequential_20/dense_82/MatMul/ReadVariableOp�-sequential_20/dense_83/BiasAdd/ReadVariableOp�,sequential_20/dense_83/MatMul/ReadVariableOp�
sequential_20/normalization/subSubnormalization_input!sequential_20_normalization_sub_y*
T0*'
_output_shapes
:���������u
 sequential_20/normalization/SqrtSqrt"sequential_20_normalization_sqrt_x*
T0*
_output_shapes

:j
%sequential_20/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
#sequential_20/normalization/MaximumMaximum$sequential_20/normalization/Sqrt:y:0.sequential_20/normalization/Maximum/y:output:0*
T0*
_output_shapes

:�
#sequential_20/normalization/truedivRealDiv#sequential_20/normalization/sub:z:0'sequential_20/normalization/Maximum:z:0*
T0*'
_output_shapes
:����������
,sequential_20/dense_80/MatMul/ReadVariableOpReadVariableOp5sequential_20_dense_80_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_20/dense_80/MatMulMatMul'sequential_20/normalization/truediv:z:04sequential_20/dense_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_20/dense_80/BiasAdd/ReadVariableOpReadVariableOp6sequential_20_dense_80_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_20/dense_80/BiasAddBiasAdd'sequential_20/dense_80/MatMul:product:05sequential_20/dense_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential_20/dense_80/ReluRelu'sequential_20/dense_80/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
!sequential_20/dropout_60/IdentityIdentity)sequential_20/dense_80/Relu:activations:0*
T0*(
_output_shapes
:�����������
,sequential_20/dense_81/MatMul/ReadVariableOpReadVariableOp5sequential_20_dense_81_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_20/dense_81/MatMulMatMul*sequential_20/dropout_60/Identity:output:04sequential_20/dense_81/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_20/dense_81/BiasAdd/ReadVariableOpReadVariableOp6sequential_20_dense_81_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_20/dense_81/BiasAddBiasAdd'sequential_20/dense_81/MatMul:product:05sequential_20/dense_81/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential_20/dense_81/ReluRelu'sequential_20/dense_81/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
!sequential_20/dropout_61/IdentityIdentity)sequential_20/dense_81/Relu:activations:0*
T0*(
_output_shapes
:�����������
,sequential_20/dense_82/MatMul/ReadVariableOpReadVariableOp5sequential_20_dense_82_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_20/dense_82/MatMulMatMul*sequential_20/dropout_61/Identity:output:04sequential_20/dense_82/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_20/dense_82/BiasAdd/ReadVariableOpReadVariableOp6sequential_20_dense_82_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_20/dense_82/BiasAddBiasAdd'sequential_20/dense_82/MatMul:product:05sequential_20/dense_82/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential_20/dense_82/ReluRelu'sequential_20/dense_82/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
!sequential_20/dropout_62/IdentityIdentity)sequential_20/dense_82/Relu:activations:0*
T0*(
_output_shapes
:�����������
,sequential_20/dense_83/MatMul/ReadVariableOpReadVariableOp5sequential_20_dense_83_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_20/dense_83/MatMulMatMul*sequential_20/dropout_62/Identity:output:04sequential_20/dense_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-sequential_20/dense_83/BiasAdd/ReadVariableOpReadVariableOp6sequential_20_dense_83_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_20/dense_83/BiasAddBiasAdd'sequential_20/dense_83/MatMul:product:05sequential_20/dense_83/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
IdentityIdentity'sequential_20/dense_83/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^sequential_20/dense_80/BiasAdd/ReadVariableOp-^sequential_20/dense_80/MatMul/ReadVariableOp.^sequential_20/dense_81/BiasAdd/ReadVariableOp-^sequential_20/dense_81/MatMul/ReadVariableOp.^sequential_20/dense_82/BiasAdd/ReadVariableOp-^sequential_20/dense_82/MatMul/ReadVariableOp.^sequential_20/dense_83/BiasAdd/ReadVariableOp-^sequential_20/dense_83/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:������������������::: : : : : : : : 2^
-sequential_20/dense_80/BiasAdd/ReadVariableOp-sequential_20/dense_80/BiasAdd/ReadVariableOp2\
,sequential_20/dense_80/MatMul/ReadVariableOp,sequential_20/dense_80/MatMul/ReadVariableOp2^
-sequential_20/dense_81/BiasAdd/ReadVariableOp-sequential_20/dense_81/BiasAdd/ReadVariableOp2\
,sequential_20/dense_81/MatMul/ReadVariableOp,sequential_20/dense_81/MatMul/ReadVariableOp2^
-sequential_20/dense_82/BiasAdd/ReadVariableOp-sequential_20/dense_82/BiasAdd/ReadVariableOp2\
,sequential_20/dense_82/MatMul/ReadVariableOp,sequential_20/dense_82/MatMul/ReadVariableOp2^
-sequential_20/dense_83/BiasAdd/ReadVariableOp-sequential_20/dense_83/BiasAdd/ReadVariableOp2\
,sequential_20/dense_83/MatMul/ReadVariableOp,sequential_20/dense_83/MatMul/ReadVariableOp:$ 

_output_shapes

::$ 

_output_shapes

::e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�	
�
__inference_loss_fn_1_56761919N
:dense_81_kernel_regularizer_l2loss_readvariableop_resource:
��
identity��1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp�
1dense_81/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:dense_81_kernel_regularizer_l2loss_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_81/kernel/Regularizer/L2LossL2Loss9dense_81/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0+dense_81/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_81/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_81/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp
�
f
H__inference_dropout_61_layer_call_and_return_conditional_losses_56761202

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
-__inference_dropout_61_layer_call_fn_56761809

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_61_layer_call_and_return_conditional_losses_56761106p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_2_56761928N
:dense_82_kernel_regularizer_l2loss_readvariableop_resource:
��
identity��1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp�
1dense_82/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:dense_82_kernel_regularizer_l2loss_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_82/kernel/Regularizer/L2LossL2Loss9dense_82/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_82/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_82/kernel/Regularizer/mulMul*dense_82/kernel/Regularizer/mul/x:output:0+dense_82/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_82/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_82/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp
�
�
F__inference_dense_82_layer_call_and_return_conditional_losses_56761123

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_82/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
1dense_82/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_82/kernel/Regularizer/L2LossL2Loss9dense_82/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_82/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_82/kernel/Regularizer/mulMul*dense_82/kernel/Regularizer/mul/x:output:0+dense_82/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_82/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_dense_81_layer_call_and_return_conditional_losses_56761088

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_81/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
1dense_81/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_81/kernel/Regularizer/L2LossL2Loss9dense_81/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0+dense_81/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_81/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
I
-__inference_dropout_60_layer_call_fn_56761763

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_60_layer_call_and_return_conditional_losses_56761191a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�'
�
__inference_adapt_step_2678822
iterator%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�IteratorGetNext�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�add/ReadVariableOp�
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*
_output_shapes

: *
output_shapes

: *
output_types
2h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeanIteratorGetNext:components:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceIteratorGetNext:components:0moments/StopGradient:output:0*
T0*
_output_shapes

: l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ^
ShapeConst*
_output_shapes
:*
dtype0	*%
valueB	"               Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: K
CastCastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_1Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: I
truedivRealDivCast:y:0
Cast_1:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0P
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:X
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:G
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0V
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype0V
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:E
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:V
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @N
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:Z
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:I
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:I
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*
validate_shape(*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22$
AssignVariableOpAssignVariableOp2"
IteratorGetNextIteratorGetNext2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22 
ReadVariableOpReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator
�>
�
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761729

inputs
normalization_sub_y
normalization_sqrt_x:
'dense_80_matmul_readvariableop_resource:	�7
(dense_80_biasadd_readvariableop_resource:	�;
'dense_81_matmul_readvariableop_resource:
��7
(dense_81_biasadd_readvariableop_resource:	�;
'dense_82_matmul_readvariableop_resource:
��7
(dense_82_biasadd_readvariableop_resource:	�:
'dense_83_matmul_readvariableop_resource:	�6
(dense_83_biasadd_readvariableop_resource:
identity��dense_80/BiasAdd/ReadVariableOp�dense_80/MatMul/ReadVariableOp�1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp�dense_81/BiasAdd/ReadVariableOp�dense_81/MatMul/ReadVariableOp�1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp�dense_82/BiasAdd/ReadVariableOp�dense_82/MatMul/ReadVariableOp�1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp�dense_83/BiasAdd/ReadVariableOp�dense_83/MatMul/ReadVariableOpg
normalization/subSubinputsnormalization_sub_y*
T0*'
_output_shapes
:���������Y
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:�
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:����������
dense_80/MatMul/ReadVariableOpReadVariableOp'dense_80_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_80/MatMulMatMulnormalization/truediv:z:0&dense_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_80/BiasAdd/ReadVariableOpReadVariableOp(dense_80_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_80/BiasAddBiasAdddense_80/MatMul:product:0'dense_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_80/ReluReludense_80/BiasAdd:output:0*
T0*(
_output_shapes
:����������o
dropout_60/IdentityIdentitydense_80/Relu:activations:0*
T0*(
_output_shapes
:�����������
dense_81/MatMul/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_81/MatMulMatMuldropout_60/Identity:output:0&dense_81/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_81/BiasAdd/ReadVariableOpReadVariableOp(dense_81_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_81/BiasAddBiasAdddense_81/MatMul:product:0'dense_81/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_81/ReluReludense_81/BiasAdd:output:0*
T0*(
_output_shapes
:����������o
dropout_61/IdentityIdentitydense_81/Relu:activations:0*
T0*(
_output_shapes
:�����������
dense_82/MatMul/ReadVariableOpReadVariableOp'dense_82_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_82/MatMulMatMuldropout_61/Identity:output:0&dense_82/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_82/BiasAdd/ReadVariableOpReadVariableOp(dense_82_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_82/BiasAddBiasAdddense_82/MatMul:product:0'dense_82/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_82/ReluReludense_82/BiasAdd:output:0*
T0*(
_output_shapes
:����������o
dropout_62/IdentityIdentitydense_82/Relu:activations:0*
T0*(
_output_shapes
:�����������
dense_83/MatMul/ReadVariableOpReadVariableOp'dense_83_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_83/MatMulMatMuldropout_62/Identity:output:0&dense_83/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_83/BiasAdd/ReadVariableOpReadVariableOp(dense_83_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_83/BiasAddBiasAdddense_83/MatMul:product:0'dense_83/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1dense_80/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_80_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"dense_80/kernel/Regularizer/L2LossL2Loss9dense_80/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0+dense_80/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_81/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_81/kernel/Regularizer/L2LossL2Loss9dense_81/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0+dense_81/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_82/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_82_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_82/kernel/Regularizer/L2LossL2Loss9dense_82/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_82/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_82/kernel/Regularizer/mulMul*dense_82/kernel/Regularizer/mul/x:output:0+dense_82/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_83/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_80/BiasAdd/ReadVariableOp^dense_80/MatMul/ReadVariableOp2^dense_80/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_81/BiasAdd/ReadVariableOp^dense_81/MatMul/ReadVariableOp2^dense_81/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_82/BiasAdd/ReadVariableOp^dense_82/MatMul/ReadVariableOp2^dense_82/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_83/BiasAdd/ReadVariableOp^dense_83/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:������������������::: : : : : : : : 2B
dense_80/BiasAdd/ReadVariableOpdense_80/BiasAdd/ReadVariableOp2@
dense_80/MatMul/ReadVariableOpdense_80/MatMul/ReadVariableOp2f
1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_81/BiasAdd/ReadVariableOpdense_81/BiasAdd/ReadVariableOp2@
dense_81/MatMul/ReadVariableOpdense_81/MatMul/ReadVariableOp2f
1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_82/BiasAdd/ReadVariableOpdense_82/BiasAdd/ReadVariableOp2@
dense_82/MatMul/ReadVariableOpdense_82/MatMul/ReadVariableOp2f
1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_83/BiasAdd/ReadVariableOpdense_83/BiasAdd/ReadVariableOp2@
dense_83/MatMul/ReadVariableOpdense_83/MatMul/ReadVariableOp:$ 

_output_shapes

::$ 

_output_shapes

::X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
0__inference_sequential_20_layer_call_fn_56761376
normalization_input
unknown
	unknown_0
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761353o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:������������������::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_output_shapes

::$ 

_output_shapes

::e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�
�
0__inference_sequential_20_layer_call_fn_56761305
normalization_input
unknown
	unknown_0
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761282o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:������������������::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_output_shapes

::$ 

_output_shapes

::e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�6
�
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761353

inputs
normalization_sub_y
normalization_sqrt_x$
dense_80_56761317:	� 
dense_80_56761319:	�%
dense_81_56761323:
�� 
dense_81_56761325:	�%
dense_82_56761329:
�� 
dense_82_56761331:	�$
dense_83_56761335:	�
dense_83_56761337:
identity�� dense_80/StatefulPartitionedCall�1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp� dense_81/StatefulPartitionedCall�1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp� dense_82/StatefulPartitionedCall�1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp� dense_83/StatefulPartitionedCallg
normalization/subSubinputsnormalization_sub_y*
T0*'
_output_shapes
:���������Y
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:�
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:����������
 dense_80/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_80_56761317dense_80_56761319*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_80_layer_call_and_return_conditional_losses_56761053�
dropout_60/PartitionedCallPartitionedCall)dense_80/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_60_layer_call_and_return_conditional_losses_56761191�
 dense_81/StatefulPartitionedCallStatefulPartitionedCall#dropout_60/PartitionedCall:output:0dense_81_56761323dense_81_56761325*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_81_layer_call_and_return_conditional_losses_56761088�
dropout_61/PartitionedCallPartitionedCall)dense_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_61_layer_call_and_return_conditional_losses_56761202�
 dense_82/StatefulPartitionedCallStatefulPartitionedCall#dropout_61/PartitionedCall:output:0dense_82_56761329dense_82_56761331*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_82_layer_call_and_return_conditional_losses_56761123�
dropout_62/PartitionedCallPartitionedCall)dense_82/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_62_layer_call_and_return_conditional_losses_56761213�
 dense_83/StatefulPartitionedCallStatefulPartitionedCall#dropout_62/PartitionedCall:output:0dense_83_56761335dense_83_56761337*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_83_layer_call_and_return_conditional_losses_56761153�
1dense_80/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_80_56761317*
_output_shapes
:	�*
dtype0�
"dense_80/kernel/Regularizer/L2LossL2Loss9dense_80/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0+dense_80/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_81/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_81_56761323* 
_output_shapes
:
��*
dtype0�
"dense_81/kernel/Regularizer/L2LossL2Loss9dense_81/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0+dense_81/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_82/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_82_56761329* 
_output_shapes
:
��*
dtype0�
"dense_82/kernel/Regularizer/L2LossL2Loss9dense_82/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_82/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_82/kernel/Regularizer/mulMul*dense_82/kernel/Regularizer/mul/x:output:0+dense_82/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_83/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_80/StatefulPartitionedCall2^dense_80/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_81/StatefulPartitionedCall2^dense_81/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_82/StatefulPartitionedCall2^dense_82/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_83/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:������������������::: : : : : : : : 2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2f
1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2f
1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2f
1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall:$ 

_output_shapes

::$ 

_output_shapes

::X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
f
H__inference_dropout_60_layer_call_and_return_conditional_losses_56761191

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_0_56761910M
:dense_80_kernel_regularizer_l2loss_readvariableop_resource:	�
identity��1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp�
1dense_80/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:dense_80_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"dense_80/kernel/Regularizer/L2LossL2Loss9dense_80/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0+dense_80/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_80/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_80/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp
�
f
H__inference_dropout_61_layer_call_and_return_conditional_losses_56761831

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
-__inference_dropout_62_layer_call_fn_56761860

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_62_layer_call_and_return_conditional_losses_56761141p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

g
H__inference_dropout_60_layer_call_and_return_conditional_losses_56761071

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�7
�
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761233
normalization_input
normalization_sub_y
normalization_sqrt_x$
dense_80_56761182:	� 
dense_80_56761184:	�%
dense_81_56761193:
�� 
dense_81_56761195:	�%
dense_82_56761204:
�� 
dense_82_56761206:	�$
dense_83_56761215:	�
dense_83_56761217:
identity�� dense_80/StatefulPartitionedCall�1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp� dense_81/StatefulPartitionedCall�1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp� dense_82/StatefulPartitionedCall�1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp� dense_83/StatefulPartitionedCallt
normalization/subSubnormalization_inputnormalization_sub_y*
T0*'
_output_shapes
:���������Y
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:�
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:����������
 dense_80/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_80_56761182dense_80_56761184*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_80_layer_call_and_return_conditional_losses_56761053�
dropout_60/PartitionedCallPartitionedCall)dense_80/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_60_layer_call_and_return_conditional_losses_56761191�
 dense_81/StatefulPartitionedCallStatefulPartitionedCall#dropout_60/PartitionedCall:output:0dense_81_56761193dense_81_56761195*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_81_layer_call_and_return_conditional_losses_56761088�
dropout_61/PartitionedCallPartitionedCall)dense_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_61_layer_call_and_return_conditional_losses_56761202�
 dense_82/StatefulPartitionedCallStatefulPartitionedCall#dropout_61/PartitionedCall:output:0dense_82_56761204dense_82_56761206*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_82_layer_call_and_return_conditional_losses_56761123�
dropout_62/PartitionedCallPartitionedCall)dense_82/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_62_layer_call_and_return_conditional_losses_56761213�
 dense_83/StatefulPartitionedCallStatefulPartitionedCall#dropout_62/PartitionedCall:output:0dense_83_56761215dense_83_56761217*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_83_layer_call_and_return_conditional_losses_56761153�
1dense_80/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_80_56761182*
_output_shapes
:	�*
dtype0�
"dense_80/kernel/Regularizer/L2LossL2Loss9dense_80/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0+dense_80/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_81/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_81_56761193* 
_output_shapes
:
��*
dtype0�
"dense_81/kernel/Regularizer/L2LossL2Loss9dense_81/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0+dense_81/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_82/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_82_56761204* 
_output_shapes
:
��*
dtype0�
"dense_82/kernel/Regularizer/L2LossL2Loss9dense_82/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_82/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_82/kernel/Regularizer/mulMul*dense_82/kernel/Regularizer/mul/x:output:0+dense_82/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_83/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_80/StatefulPartitionedCall2^dense_80/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_81/StatefulPartitionedCall2^dense_81/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_82/StatefulPartitionedCall2^dense_82/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_83/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:������������������::: : : : : : : : 2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2f
1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2f
1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2f
1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall:$ 

_output_shapes

::$ 

_output_shapes

::e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
��
�
$__inference__traced_restore_56762242
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:$
assignvariableop_2_count_1:	 5
"assignvariableop_3_dense_80_kernel:	�/
 assignvariableop_4_dense_80_bias:	�6
"assignvariableop_5_dense_81_kernel:
��/
 assignvariableop_6_dense_81_bias:	�6
"assignvariableop_7_dense_82_kernel:
��/
 assignvariableop_8_dense_82_bias:	�5
"assignvariableop_9_dense_83_kernel:	�/
!assignvariableop_10_dense_83_bias:'
assignvariableop_11_iteration:	 +
!assignvariableop_12_learning_rate: =
*assignvariableop_13_adam_m_dense_80_kernel:	�=
*assignvariableop_14_adam_v_dense_80_kernel:	�7
(assignvariableop_15_adam_m_dense_80_bias:	�7
(assignvariableop_16_adam_v_dense_80_bias:	�>
*assignvariableop_17_adam_m_dense_81_kernel:
��>
*assignvariableop_18_adam_v_dense_81_kernel:
��7
(assignvariableop_19_adam_m_dense_81_bias:	�7
(assignvariableop_20_adam_v_dense_81_bias:	�>
*assignvariableop_21_adam_m_dense_82_kernel:
��>
*assignvariableop_22_adam_v_dense_82_kernel:
��7
(assignvariableop_23_adam_m_dense_82_bias:	�7
(assignvariableop_24_adam_v_dense_82_bias:	�=
*assignvariableop_25_adam_m_dense_83_kernel:	�=
*assignvariableop_26_adam_v_dense_83_kernel:	�6
(assignvariableop_27_adam_m_dense_83_bias:6
(assignvariableop_28_adam_v_dense_83_bias:#
assignvariableop_29_total: #
assignvariableop_30_count: 
identity_32��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::*.
dtypes$
"2 		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_meanIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_varianceIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_count_1Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_80_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_80_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_81_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense_81_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_82_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense_82_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dense_83_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dense_83_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_iterationIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp!assignvariableop_12_learning_rateIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_m_dense_80_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_v_dense_80_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_m_dense_80_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_v_dense_80_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_m_dense_81_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_v_dense_81_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_m_dense_81_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_v_dense_81_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_m_dense_82_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_v_dense_82_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_m_dense_82_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_v_dense_82_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_m_dense_83_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_v_dense_83_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_m_dense_83_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_v_dense_83_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

g
H__inference_dropout_62_layer_call_and_return_conditional_losses_56761141

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_dense_80_layer_call_and_return_conditional_losses_56761753

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_80/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
1dense_80/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"dense_80/kernel/Regularizer/L2LossL2Loss9dense_80/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0+dense_80/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_80/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
0__inference_sequential_20_layer_call_fn_56761577

inputs
unknown
	unknown_0
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761282o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:������������������::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_output_shapes

::$ 

_output_shapes

::X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
f
H__inference_dropout_62_layer_call_and_return_conditional_losses_56761213

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�;
�
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761282

inputs
normalization_sub_y
normalization_sqrt_x$
dense_80_56761246:	� 
dense_80_56761248:	�%
dense_81_56761252:
�� 
dense_81_56761254:	�%
dense_82_56761258:
�� 
dense_82_56761260:	�$
dense_83_56761264:	�
dense_83_56761266:
identity�� dense_80/StatefulPartitionedCall�1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp� dense_81/StatefulPartitionedCall�1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp� dense_82/StatefulPartitionedCall�1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp� dense_83/StatefulPartitionedCall�"dropout_60/StatefulPartitionedCall�"dropout_61/StatefulPartitionedCall�"dropout_62/StatefulPartitionedCallg
normalization/subSubinputsnormalization_sub_y*
T0*'
_output_shapes
:���������Y
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:�
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:����������
 dense_80/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_80_56761246dense_80_56761248*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_80_layer_call_and_return_conditional_losses_56761053�
"dropout_60/StatefulPartitionedCallStatefulPartitionedCall)dense_80/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_60_layer_call_and_return_conditional_losses_56761071�
 dense_81/StatefulPartitionedCallStatefulPartitionedCall+dropout_60/StatefulPartitionedCall:output:0dense_81_56761252dense_81_56761254*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_81_layer_call_and_return_conditional_losses_56761088�
"dropout_61/StatefulPartitionedCallStatefulPartitionedCall)dense_81/StatefulPartitionedCall:output:0#^dropout_60/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_61_layer_call_and_return_conditional_losses_56761106�
 dense_82/StatefulPartitionedCallStatefulPartitionedCall+dropout_61/StatefulPartitionedCall:output:0dense_82_56761258dense_82_56761260*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_82_layer_call_and_return_conditional_losses_56761123�
"dropout_62/StatefulPartitionedCallStatefulPartitionedCall)dense_82/StatefulPartitionedCall:output:0#^dropout_61/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_62_layer_call_and_return_conditional_losses_56761141�
 dense_83/StatefulPartitionedCallStatefulPartitionedCall+dropout_62/StatefulPartitionedCall:output:0dense_83_56761264dense_83_56761266*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_83_layer_call_and_return_conditional_losses_56761153�
1dense_80/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_80_56761246*
_output_shapes
:	�*
dtype0�
"dense_80/kernel/Regularizer/L2LossL2Loss9dense_80/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0+dense_80/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_81/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_81_56761252* 
_output_shapes
:
��*
dtype0�
"dense_81/kernel/Regularizer/L2LossL2Loss9dense_81/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0+dense_81/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_82/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_82_56761258* 
_output_shapes
:
��*
dtype0�
"dense_82/kernel/Regularizer/L2LossL2Loss9dense_82/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_82/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_82/kernel/Regularizer/mulMul*dense_82/kernel/Regularizer/mul/x:output:0+dense_82/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_83/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_80/StatefulPartitionedCall2^dense_80/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_81/StatefulPartitionedCall2^dense_81/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_82/StatefulPartitionedCall2^dense_82/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_83/StatefulPartitionedCall#^dropout_60/StatefulPartitionedCall#^dropout_61/StatefulPartitionedCall#^dropout_62/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:������������������::: : : : : : : : 2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2f
1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2f
1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp1dense_81/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_82/StatefulPartitionedCall dense_82/StatefulPartitionedCall2f
1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_83/StatefulPartitionedCall dense_83/StatefulPartitionedCall2H
"dropout_60/StatefulPartitionedCall"dropout_60/StatefulPartitionedCall2H
"dropout_61/StatefulPartitionedCall"dropout_61/StatefulPartitionedCall2H
"dropout_62/StatefulPartitionedCall"dropout_62/StatefulPartitionedCall:$ 

_output_shapes

::$ 

_output_shapes

::X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
F__inference_dense_80_layer_call_and_return_conditional_losses_56761053

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_80/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
1dense_80/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"dense_80/kernel/Regularizer/L2LossL2Loss9dense_80/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0+dense_80/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_80/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp1dense_80/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
&__inference_signature_wrapper_56761540
normalization_input
unknown
	unknown_0
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_56761027o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:������������������::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_output_shapes

::$ 

_output_shapes

::e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�
�
+__inference_dense_83_layer_call_fn_56761891

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_83_layer_call_and_return_conditional_losses_56761153o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
I
-__inference_dropout_62_layer_call_fn_56761865

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_62_layer_call_and_return_conditional_losses_56761213a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
H__inference_dropout_62_layer_call_and_return_conditional_losses_56761882

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_dense_80_layer_call_fn_56761738

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_80_layer_call_and_return_conditional_losses_56761053p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_dense_82_layer_call_and_return_conditional_losses_56761855

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_82/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:�����������
1dense_82/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_82/kernel/Regularizer/L2LossL2Loss9dense_82/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_82/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
dense_82/kernel/Regularizer/mulMul*dense_82/kernel/Regularizer/mul/x:output:0+dense_82/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_82/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp1dense_82/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
F__inference_dense_83_layer_call_and_return_conditional_losses_56761901

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
H__inference_dropout_60_layer_call_and_return_conditional_losses_56761780

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_dense_82_layer_call_fn_56761840

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_82_layer_call_and_return_conditional_losses_56761123p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
\
normalization_inputE
%serving_default_normalization_input:0������������������<
dense_830
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
_adapt_function"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias"
_tf_keras_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)_random_generator"
_tf_keras_layer
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

0kernel
1bias"
_tf_keras_layer
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
8_random_generator"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias"
_tf_keras_layer
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
G_random_generator"
_tf_keras_layer
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias"
_tf_keras_layer
n
0
1
2
!3
"4
05
16
?7
@8
N9
O10"
trackable_list_wrapper
X
!0
"1
02
13
?4
@5
N6
O7"
trackable_list_wrapper
5
P0
Q1
R2"
trackable_list_wrapper
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Xtrace_0
Ytrace_1
Ztrace_2
[trace_32�
0__inference_sequential_20_layer_call_fn_56761305
0__inference_sequential_20_layer_call_fn_56761376
0__inference_sequential_20_layer_call_fn_56761577
0__inference_sequential_20_layer_call_fn_56761602�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zXtrace_0zYtrace_1zZtrace_2z[trace_3
�
\trace_0
]trace_1
^trace_2
_trace_32�
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761172
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761233
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761676
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761729�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z\trace_0z]trace_1z^trace_2z_trace_3
�
`	capture_0
a	capture_1B�
#__inference__wrapped_model_56761027normalization_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`	capture_0za	capture_1
�
b
_variables
c_iterations
d_learning_rate
e_index_dict
f
_momentums
g_velocities
h_update_step_xla"
experimentalOptimizer
,
iserving_default"
signature_map
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
�
jtrace_02�
__inference_adapt_step_2678822�
���
FullArgSpec
args�

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zjtrace_0
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
'
P0"
trackable_list_wrapper
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
�
ptrace_02�
+__inference_dense_80_layer_call_fn_56761738�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zptrace_0
�
qtrace_02�
F__inference_dense_80_layer_call_and_return_conditional_losses_56761753�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zqtrace_0
": 	�2dense_80/kernel
:�2dense_80/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
wtrace_0
xtrace_12�
-__inference_dropout_60_layer_call_fn_56761758
-__inference_dropout_60_layer_call_fn_56761763�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zwtrace_0zxtrace_1
�
ytrace_0
ztrace_12�
H__inference_dropout_60_layer_call_and_return_conditional_losses_56761775
H__inference_dropout_60_layer_call_and_return_conditional_losses_56761780�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zytrace_0zztrace_1
"
_generic_user_object
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
�
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_81_layer_call_fn_56761789�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_dense_81_layer_call_and_return_conditional_losses_56761804�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!
��2dense_81/kernel
:�2dense_81/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
-__inference_dropout_61_layer_call_fn_56761809
-__inference_dropout_61_layer_call_fn_56761814�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
H__inference_dropout_61_layer_call_and_return_conditional_losses_56761826
H__inference_dropout_61_layer_call_and_return_conditional_losses_56761831�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
'
R0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_82_layer_call_fn_56761840�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_dense_82_layer_call_and_return_conditional_losses_56761855�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!
��2dense_82/kernel
:�2dense_82/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
-__inference_dropout_62_layer_call_fn_56761860
-__inference_dropout_62_layer_call_fn_56761865�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
H__inference_dropout_62_layer_call_and_return_conditional_losses_56761877
H__inference_dropout_62_layer_call_and_return_conditional_losses_56761882�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_83_layer_call_fn_56761891�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_dense_83_layer_call_and_return_conditional_losses_56761901�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 	�2dense_83/kernel
:2dense_83/bias
�
�trace_02�
__inference_loss_fn_0_56761910�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_56761919�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_56761928�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
5
0
1
2"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
`	capture_0
a	capture_1B�
0__inference_sequential_20_layer_call_fn_56761305normalization_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`	capture_0za	capture_1
�
`	capture_0
a	capture_1B�
0__inference_sequential_20_layer_call_fn_56761376normalization_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`	capture_0za	capture_1
�
`	capture_0
a	capture_1B�
0__inference_sequential_20_layer_call_fn_56761577inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`	capture_0za	capture_1
�
`	capture_0
a	capture_1B�
0__inference_sequential_20_layer_call_fn_56761602inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`	capture_0za	capture_1
�
`	capture_0
a	capture_1B�
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761172normalization_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`	capture_0za	capture_1
�
`	capture_0
a	capture_1B�
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761233normalization_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`	capture_0za	capture_1
�
`	capture_0
a	capture_1B�
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761676inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`	capture_0za	capture_1
�
`	capture_0
a	capture_1B�
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761729inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`	capture_0za	capture_1
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
�
c0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�
`	capture_0
a	capture_1B�
&__inference_signature_wrapper_56761540normalization_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`	capture_0za	capture_1
�B�
__inference_adapt_step_2678822iterator"�
���
FullArgSpec
args�

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
P0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_80_layer_call_fn_56761738inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_80_layer_call_and_return_conditional_losses_56761753inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
-__inference_dropout_60_layer_call_fn_56761758inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_dropout_60_layer_call_fn_56761763inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_60_layer_call_and_return_conditional_losses_56761775inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_60_layer_call_and_return_conditional_losses_56761780inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_81_layer_call_fn_56761789inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_81_layer_call_and_return_conditional_losses_56761804inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
-__inference_dropout_61_layer_call_fn_56761809inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_dropout_61_layer_call_fn_56761814inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_61_layer_call_and_return_conditional_losses_56761826inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_61_layer_call_and_return_conditional_losses_56761831inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
R0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_82_layer_call_fn_56761840inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_82_layer_call_and_return_conditional_losses_56761855inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
-__inference_dropout_62_layer_call_fn_56761860inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_dropout_62_layer_call_fn_56761865inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_62_layer_call_and_return_conditional_losses_56761877inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_62_layer_call_and_return_conditional_losses_56761882inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
+__inference_dense_83_layer_call_fn_56761891inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_83_layer_call_and_return_conditional_losses_56761901inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_56761910"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_56761919"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_2_56761928"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
':%	�2Adam/m/dense_80/kernel
':%	�2Adam/v/dense_80/kernel
!:�2Adam/m/dense_80/bias
!:�2Adam/v/dense_80/bias
(:&
��2Adam/m/dense_81/kernel
(:&
��2Adam/v/dense_81/kernel
!:�2Adam/m/dense_81/bias
!:�2Adam/v/dense_81/bias
(:&
��2Adam/m/dense_82/kernel
(:&
��2Adam/v/dense_82/kernel
!:�2Adam/m/dense_82/bias
!:�2Adam/v/dense_82/bias
':%	�2Adam/m/dense_83/kernel
':%	�2Adam/v/dense_83/kernel
 :2Adam/m/dense_83/bias
 :2Adam/v/dense_83/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count�
#__inference__wrapped_model_56761027�
`a!"01?@NOE�B
;�8
6�3
normalization_input������������������
� "3�0
.
dense_83"�
dense_83���������g
__inference_adapt_step_2678822E:�7
0�-
+�(�
� IteratorSpec 
� "
 �
F__inference_dense_80_layer_call_and_return_conditional_losses_56761753d!"/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
+__inference_dense_80_layer_call_fn_56761738Y!"/�,
%�"
 �
inputs���������
� ""�
unknown�����������
F__inference_dense_81_layer_call_and_return_conditional_losses_56761804e010�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
+__inference_dense_81_layer_call_fn_56761789Z010�-
&�#
!�
inputs����������
� ""�
unknown�����������
F__inference_dense_82_layer_call_and_return_conditional_losses_56761855e?@0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
+__inference_dense_82_layer_call_fn_56761840Z?@0�-
&�#
!�
inputs����������
� ""�
unknown�����������
F__inference_dense_83_layer_call_and_return_conditional_losses_56761901dNO0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
+__inference_dense_83_layer_call_fn_56761891YNO0�-
&�#
!�
inputs����������
� "!�
unknown����������
H__inference_dropout_60_layer_call_and_return_conditional_losses_56761775e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
H__inference_dropout_60_layer_call_and_return_conditional_losses_56761780e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
-__inference_dropout_60_layer_call_fn_56761758Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
-__inference_dropout_60_layer_call_fn_56761763Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
H__inference_dropout_61_layer_call_and_return_conditional_losses_56761826e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
H__inference_dropout_61_layer_call_and_return_conditional_losses_56761831e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
-__inference_dropout_61_layer_call_fn_56761809Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
-__inference_dropout_61_layer_call_fn_56761814Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
H__inference_dropout_62_layer_call_and_return_conditional_losses_56761877e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
H__inference_dropout_62_layer_call_and_return_conditional_losses_56761882e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
-__inference_dropout_62_layer_call_fn_56761860Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
-__inference_dropout_62_layer_call_fn_56761865Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown����������F
__inference_loss_fn_0_56761910$!�

� 
� "�
unknown F
__inference_loss_fn_1_56761919$0�

� 
� "�
unknown F
__inference_loss_fn_2_56761928$?�

� 
� "�
unknown �
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761172�
`a!"01?@NOM�J
C�@
6�3
normalization_input������������������
p

 
� ",�)
"�
tensor_0���������
� �
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761233�
`a!"01?@NOM�J
C�@
6�3
normalization_input������������������
p 

 
� ",�)
"�
tensor_0���������
� �
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761676|
`a!"01?@NO@�=
6�3
)�&
inputs������������������
p

 
� ",�)
"�
tensor_0���������
� �
K__inference_sequential_20_layer_call_and_return_conditional_losses_56761729|
`a!"01?@NO@�=
6�3
)�&
inputs������������������
p 

 
� ",�)
"�
tensor_0���������
� �
0__inference_sequential_20_layer_call_fn_56761305~
`a!"01?@NOM�J
C�@
6�3
normalization_input������������������
p

 
� "!�
unknown����������
0__inference_sequential_20_layer_call_fn_56761376~
`a!"01?@NOM�J
C�@
6�3
normalization_input������������������
p 

 
� "!�
unknown����������
0__inference_sequential_20_layer_call_fn_56761577q
`a!"01?@NO@�=
6�3
)�&
inputs������������������
p

 
� "!�
unknown����������
0__inference_sequential_20_layer_call_fn_56761602q
`a!"01?@NO@�=
6�3
)�&
inputs������������������
p 

 
� "!�
unknown����������
&__inference_signature_wrapper_56761540�
`a!"01?@NO\�Y
� 
R�O
M
normalization_input6�3
normalization_input������������������"3�0
.
dense_83"�
dense_83���������