��	
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
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
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
Adam/v/dense_123/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/v/dense_123/bias
{
)Adam/v/dense_123/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_123/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_123/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/m/dense_123/bias
{
)Adam/m/dense_123/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_123/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_123/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/v/dense_123/kernel
�
+Adam/v/dense_123/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_123/kernel*
_output_shapes

: *
dtype0
�
Adam/m/dense_123/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/m/dense_123/kernel
�
+Adam/m/dense_123/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_123/kernel*
_output_shapes

: *
dtype0
�
Adam/v/dense_122/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/v/dense_122/bias
{
)Adam/v/dense_122/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_122/bias*
_output_shapes
: *
dtype0
�
Adam/m/dense_122/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/m/dense_122/bias
{
)Adam/m/dense_122/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_122/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_122/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/v/dense_122/kernel
�
+Adam/v/dense_122/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_122/kernel*
_output_shapes

:  *
dtype0
�
Adam/m/dense_122/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/m/dense_122/kernel
�
+Adam/m/dense_122/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_122/kernel*
_output_shapes

:  *
dtype0
�
Adam/v/dense_121/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/v/dense_121/bias
{
)Adam/v/dense_121/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_121/bias*
_output_shapes
: *
dtype0
�
Adam/m/dense_121/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/m/dense_121/bias
{
)Adam/m/dense_121/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_121/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_121/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/v/dense_121/kernel
�
+Adam/v/dense_121/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_121/kernel*
_output_shapes

: *
dtype0
�
Adam/m/dense_121/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameAdam/m/dense_121/kernel
�
+Adam/m/dense_121/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_121/kernel*
_output_shapes

: *
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
t
dense_123/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_123/bias
m
"dense_123/bias/Read/ReadVariableOpReadVariableOpdense_123/bias*
_output_shapes
:*
dtype0
|
dense_123/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_namedense_123/kernel
u
$dense_123/kernel/Read/ReadVariableOpReadVariableOpdense_123/kernel*
_output_shapes

: *
dtype0
t
dense_122/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_122/bias
m
"dense_122/bias/Read/ReadVariableOpReadVariableOpdense_122/bias*
_output_shapes
: *
dtype0
|
dense_122/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_122/kernel
u
$dense_122/kernel/Read/ReadVariableOpReadVariableOpdense_122/kernel*
_output_shapes

:  *
dtype0
t
dense_121/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_121/bias
m
"dense_121/bias/Read/ReadVariableOpReadVariableOpdense_121/bias*
_output_shapes
: *
dtype0
|
dense_121/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_namedense_121/kernel
u
$dense_121/kernel/Read/ReadVariableOpReadVariableOpdense_121/kernel*
_output_shapes

: *
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
StatefulPartitionedCallStatefulPartitionedCall#serving_default_normalization_inputConst_1Constdense_121/kerneldense_121/biasdense_122/kerneldense_122/biasdense_123/kerneldense_123/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *0
f+R)
'__inference_signature_wrapper_266407977

NoOpNoOp
�:
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*�9
value�9B�9 B�9
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
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
_adapt_function*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias*
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_random_generator* 
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias*
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_random_generator* 
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias*
C
0
1
2
3
 4
.5
/6
=7
>8*
.
0
 1
.2
/3
=4
>5*

?0
@1* 
�
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ftrace_0
Gtrace_1
Htrace_2
Itrace_3* 
6
Jtrace_0
Ktrace_1
Ltrace_2
Mtrace_3* 
 
N	capture_0
O	capture_1* 
�
P
_variables
Q_iterations
R_learning_rate
S_index_dict
T
_momentums
U_velocities
V_update_step_xla*

Wserving_default* 
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
Xtrace_0* 

0
 1*

0
 1*
	
?0* 
�
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

^trace_0* 

_trace_0* 
`Z
VARIABLE_VALUEdense_121/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_121/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses* 

etrace_0
ftrace_1* 

gtrace_0
htrace_1* 
* 

.0
/1*

.0
/1*
	
@0* 
�
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

ntrace_0* 

otrace_0* 
`Z
VARIABLE_VALUEdense_122/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_122/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 

utrace_0
vtrace_1* 

wtrace_0
xtrace_1* 
* 

=0
>1*

=0
>1*
* 
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

~trace_0* 

trace_0* 
`Z
VARIABLE_VALUEdense_123/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_123/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 

0
1
2*
.
0
1
2
3
4
5*

�0*
* 
* 
 
N	capture_0
O	capture_1* 
 
N	capture_0
O	capture_1* 
 
N	capture_0
O	capture_1* 
 
N	capture_0
O	capture_1* 
 
N	capture_0
O	capture_1* 
 
N	capture_0
O	capture_1* 
 
N	capture_0
O	capture_1* 
 
N	capture_0
O	capture_1* 
* 
* 
n
Q0
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
�12*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
4
�0
�1
�2
�3
�4
�5*
4
�0
�1
�2
�3
�4
�5*
* 
 
N	capture_0
O	capture_1* 
* 
* 
* 
* 
	
?0* 
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
	
@0* 
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
b\
VARIABLE_VALUEAdam/m/dense_121/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_121/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_121/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_121/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_122/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_122/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_122/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_122/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_123/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/dense_123/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_123/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_123/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
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
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemeanvariancecount_1dense_121/kerneldense_121/biasdense_122/kerneldense_122/biasdense_123/kerneldense_123/bias	iterationlearning_rateAdam/m/dense_121/kernelAdam/v/dense_121/kernelAdam/m/dense_121/biasAdam/v/dense_121/biasAdam/m/dense_122/kernelAdam/v/dense_122/kernelAdam/m/dense_122/biasAdam/v/dense_122/biasAdam/m/dense_123/kernelAdam/v/dense_123/kernelAdam/m/dense_123/biasAdam/v/dense_123/biastotalcountConst_2*&
Tin
2*
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
GPU 2J 8� *+
f&R$
"__inference__traced_save_266408437
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecount_1dense_121/kerneldense_121/biasdense_122/kerneldense_122/biasdense_123/kerneldense_123/bias	iterationlearning_rateAdam/m/dense_121/kernelAdam/v/dense_121/kernelAdam/m/dense_121/biasAdam/v/dense_121/biasAdam/m/dense_122/kernelAdam/v/dense_122/kernelAdam/m/dense_122/biasAdam/v/dense_122/biasAdam/m/dense_123/kernelAdam/v/dense_123/kernelAdam/m/dense_123/biasAdam/v/dense_123/biastotalcount*%
Tin
2*
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
GPU 2J 8� *.
f)R'
%__inference__traced_restore_266408522З
�

�
1__inference_sequential_56_layer_call_fn_266408006

inputs
unknown
	unknown_0
	unknown_1: 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_56_layer_call_and_return_conditional_losses_266407774o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������::: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_output_shapes

::$ 

_output_shapes

::X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
g
.__inference_dropout_65_layer_call_fn_266408152

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dropout_65_layer_call_and_return_conditional_losses_266407627o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
g
I__inference_dropout_66_layer_call_and_return_conditional_losses_266407719

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�-
�
L__inference_sequential_56_layer_call_and_return_conditional_losses_266407689
normalization_input
normalization_sub_y
normalization_sqrt_x%
dense_121_266407610: !
dense_121_266407612: %
dense_122_266407645:  !
dense_122_266407647: %
dense_123_266407675: !
dense_123_266407677:
identity��!dense_121/StatefulPartitionedCall�2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp�!dense_122/StatefulPartitionedCall�2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp�!dense_123/StatefulPartitionedCall�"dropout_65/StatefulPartitionedCall�"dropout_66/StatefulPartitionedCallt
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
!dense_121/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_121_266407610dense_121_266407612*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_121_layer_call_and_return_conditional_losses_266407609�
"dropout_65/StatefulPartitionedCallStatefulPartitionedCall*dense_121/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dropout_65_layer_call_and_return_conditional_losses_266407627�
!dense_122/StatefulPartitionedCallStatefulPartitionedCall+dropout_65/StatefulPartitionedCall:output:0dense_122_266407645dense_122_266407647*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_122_layer_call_and_return_conditional_losses_266407644�
"dropout_66/StatefulPartitionedCallStatefulPartitionedCall*dense_122/StatefulPartitionedCall:output:0#^dropout_65/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dropout_66_layer_call_and_return_conditional_losses_266407662�
!dense_123/StatefulPartitionedCallStatefulPartitionedCall+dropout_66/StatefulPartitionedCall:output:0dense_123_266407675dense_123_266407677*
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
GPU 2J 8� *Q
fLRJ
H__inference_dense_123_layer_call_and_return_conditional_losses_266407674�
2dense_121/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_121_266407610*
_output_shapes

: *
dtype0�
#dense_121/kernel/Regularizer/L2LossL2Loss:dense_121/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_121/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
 dense_121/kernel/Regularizer/mulMul+dense_121/kernel/Regularizer/mul/x:output:0,dense_121/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_122/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_122_266407645*
_output_shapes

:  *
dtype0�
#dense_122/kernel/Regularizer/L2LossL2Loss:dense_122/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0,dense_122/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_123/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_121/StatefulPartitionedCall3^dense_121/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_122/StatefulPartitionedCall3^dense_122/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_123/StatefulPartitionedCall#^dropout_65/StatefulPartitionedCall#^dropout_66/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������::: : : : : : 2F
!dense_121/StatefulPartitionedCall!dense_121/StatefulPartitionedCall2h
2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2h
2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_123/StatefulPartitionedCall!dense_123/StatefulPartitionedCall2H
"dropout_65/StatefulPartitionedCall"dropout_65/StatefulPartitionedCall2H
"dropout_66/StatefulPartitionedCall"dropout_66/StatefulPartitionedCall:$ 

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
__inference_loss_fn_1_266408262M
;dense_122_kernel_regularizer_l2loss_readvariableop_resource:  
identity��2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp�
2dense_122/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp;dense_122_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:  *
dtype0�
#dense_122/kernel/Regularizer/L2LossL2Loss:dense_122/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0,dense_122/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_122/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^dense_122/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp
�
�
H__inference_dense_122_layer_call_and_return_conditional_losses_266408198

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2dense_122/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� �
2dense_122/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
#dense_122/kernel/Regularizer/L2LossL2Loss:dense_122/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0,dense_122/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_122/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�'
�
__inference_adapt_step_2715533
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
�/
�
L__inference_sequential_56_layer_call_and_return_conditional_losses_266408123

inputs
normalization_sub_y
normalization_sqrt_x:
(dense_121_matmul_readvariableop_resource: 7
)dense_121_biasadd_readvariableop_resource: :
(dense_122_matmul_readvariableop_resource:  7
)dense_122_biasadd_readvariableop_resource: :
(dense_123_matmul_readvariableop_resource: 7
)dense_123_biasadd_readvariableop_resource:
identity�� dense_121/BiasAdd/ReadVariableOp�dense_121/MatMul/ReadVariableOp�2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp� dense_122/BiasAdd/ReadVariableOp�dense_122/MatMul/ReadVariableOp�2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp� dense_123/BiasAdd/ReadVariableOp�dense_123/MatMul/ReadVariableOpg
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
dense_121/MatMul/ReadVariableOpReadVariableOp(dense_121_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_121/MatMulMatMulnormalization/truediv:z:0'dense_121/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
 dense_121/BiasAdd/ReadVariableOpReadVariableOp)dense_121_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_121/BiasAddBiasAdddense_121/MatMul:product:0(dense_121/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� d
dense_121/ReluReludense_121/BiasAdd:output:0*
T0*'
_output_shapes
:��������� o
dropout_65/IdentityIdentitydense_121/Relu:activations:0*
T0*'
_output_shapes
:��������� �
dense_122/MatMul/ReadVariableOpReadVariableOp(dense_122_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_122/MatMulMatMuldropout_65/Identity:output:0'dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
 dense_122/BiasAdd/ReadVariableOpReadVariableOp)dense_122_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_122/BiasAddBiasAdddense_122/MatMul:product:0(dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� d
dense_122/ReluReludense_122/BiasAdd:output:0*
T0*'
_output_shapes
:��������� o
dropout_66/IdentityIdentitydense_122/Relu:activations:0*
T0*'
_output_shapes
:��������� �
dense_123/MatMul/ReadVariableOpReadVariableOp(dense_123_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_123/MatMulMatMuldropout_66/Identity:output:0'dense_123/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_123/BiasAdd/ReadVariableOpReadVariableOp)dense_123_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_123/BiasAddBiasAdddense_123/MatMul:product:0(dense_123/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2dense_121/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_121_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
#dense_121/kernel/Regularizer/L2LossL2Loss:dense_121/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_121/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
 dense_121/kernel/Regularizer/mulMul+dense_121/kernel/Regularizer/mul/x:output:0,dense_121/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_122/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_122_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
#dense_122/kernel/Regularizer/L2LossL2Loss:dense_122/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0,dense_122/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_123/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_121/BiasAdd/ReadVariableOp ^dense_121/MatMul/ReadVariableOp3^dense_121/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_122/BiasAdd/ReadVariableOp ^dense_122/MatMul/ReadVariableOp3^dense_122/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_123/BiasAdd/ReadVariableOp ^dense_123/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������::: : : : : : 2D
 dense_121/BiasAdd/ReadVariableOp dense_121/BiasAdd/ReadVariableOp2B
dense_121/MatMul/ReadVariableOpdense_121/MatMul/ReadVariableOp2h
2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_122/BiasAdd/ReadVariableOp dense_122/BiasAdd/ReadVariableOp2B
dense_122/MatMul/ReadVariableOpdense_122/MatMul/ReadVariableOp2h
2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_123/BiasAdd/ReadVariableOp dense_123/BiasAdd/ReadVariableOp2B
dense_123/MatMul/ReadVariableOpdense_123/MatMul/ReadVariableOp:$ 

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
H__inference_dense_122_layer_call_and_return_conditional_losses_266407644

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2dense_122/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� �
2dense_122/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
#dense_122/kernel/Regularizer/L2LossL2Loss:dense_122/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0,dense_122/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_122/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_0_266408253M
;dense_121_kernel_regularizer_l2loss_readvariableop_resource: 
identity��2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp�
2dense_121/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp;dense_121_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

: *
dtype0�
#dense_121/kernel/Regularizer/L2LossL2Loss:dense_121/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_121/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
 dense_121/kernel/Regularizer/mulMul+dense_121/kernel/Regularizer/mul/x:output:0,dense_121/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_121/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^dense_121/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp
�

h
I__inference_dropout_66_layer_call_and_return_conditional_losses_266408220

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
-__inference_dense_122_layer_call_fn_266408183

inputs
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_122_layer_call_and_return_conditional_losses_266407644o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
H__inference_dense_123_layer_call_and_return_conditional_losses_266408244

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
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
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
J
.__inference_dropout_65_layer_call_fn_266408157

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dropout_65_layer_call_and_return_conditional_losses_266407708`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

h
I__inference_dropout_65_layer_call_and_return_conditional_losses_266408169

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�,
�
$__inference__wrapped_model_266407583
normalization_input%
!sequential_56_normalization_sub_y&
"sequential_56_normalization_sqrt_xH
6sequential_56_dense_121_matmul_readvariableop_resource: E
7sequential_56_dense_121_biasadd_readvariableop_resource: H
6sequential_56_dense_122_matmul_readvariableop_resource:  E
7sequential_56_dense_122_biasadd_readvariableop_resource: H
6sequential_56_dense_123_matmul_readvariableop_resource: E
7sequential_56_dense_123_biasadd_readvariableop_resource:
identity��.sequential_56/dense_121/BiasAdd/ReadVariableOp�-sequential_56/dense_121/MatMul/ReadVariableOp�.sequential_56/dense_122/BiasAdd/ReadVariableOp�-sequential_56/dense_122/MatMul/ReadVariableOp�.sequential_56/dense_123/BiasAdd/ReadVariableOp�-sequential_56/dense_123/MatMul/ReadVariableOp�
sequential_56/normalization/subSubnormalization_input!sequential_56_normalization_sub_y*
T0*'
_output_shapes
:���������u
 sequential_56/normalization/SqrtSqrt"sequential_56_normalization_sqrt_x*
T0*
_output_shapes

:j
%sequential_56/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
#sequential_56/normalization/MaximumMaximum$sequential_56/normalization/Sqrt:y:0.sequential_56/normalization/Maximum/y:output:0*
T0*
_output_shapes

:�
#sequential_56/normalization/truedivRealDiv#sequential_56/normalization/sub:z:0'sequential_56/normalization/Maximum:z:0*
T0*'
_output_shapes
:����������
-sequential_56/dense_121/MatMul/ReadVariableOpReadVariableOp6sequential_56_dense_121_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
sequential_56/dense_121/MatMulMatMul'sequential_56/normalization/truediv:z:05sequential_56/dense_121/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
.sequential_56/dense_121/BiasAdd/ReadVariableOpReadVariableOp7sequential_56_dense_121_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential_56/dense_121/BiasAddBiasAdd(sequential_56/dense_121/MatMul:product:06sequential_56/dense_121/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
sequential_56/dense_121/ReluRelu(sequential_56/dense_121/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
!sequential_56/dropout_65/IdentityIdentity*sequential_56/dense_121/Relu:activations:0*
T0*'
_output_shapes
:��������� �
-sequential_56/dense_122/MatMul/ReadVariableOpReadVariableOp6sequential_56_dense_122_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
sequential_56/dense_122/MatMulMatMul*sequential_56/dropout_65/Identity:output:05sequential_56/dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
.sequential_56/dense_122/BiasAdd/ReadVariableOpReadVariableOp7sequential_56_dense_122_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential_56/dense_122/BiasAddBiasAdd(sequential_56/dense_122/MatMul:product:06sequential_56/dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
sequential_56/dense_122/ReluRelu(sequential_56/dense_122/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
!sequential_56/dropout_66/IdentityIdentity*sequential_56/dense_122/Relu:activations:0*
T0*'
_output_shapes
:��������� �
-sequential_56/dense_123/MatMul/ReadVariableOpReadVariableOp6sequential_56_dense_123_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
sequential_56/dense_123/MatMulMatMul*sequential_56/dropout_66/Identity:output:05sequential_56/dense_123/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.sequential_56/dense_123/BiasAdd/ReadVariableOpReadVariableOp7sequential_56_dense_123_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_56/dense_123/BiasAddBiasAdd(sequential_56/dense_123/MatMul:product:06sequential_56/dense_123/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w
IdentityIdentity(sequential_56/dense_123/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^sequential_56/dense_121/BiasAdd/ReadVariableOp.^sequential_56/dense_121/MatMul/ReadVariableOp/^sequential_56/dense_122/BiasAdd/ReadVariableOp.^sequential_56/dense_122/MatMul/ReadVariableOp/^sequential_56/dense_123/BiasAdd/ReadVariableOp.^sequential_56/dense_123/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������::: : : : : : 2`
.sequential_56/dense_121/BiasAdd/ReadVariableOp.sequential_56/dense_121/BiasAdd/ReadVariableOp2^
-sequential_56/dense_121/MatMul/ReadVariableOp-sequential_56/dense_121/MatMul/ReadVariableOp2`
.sequential_56/dense_122/BiasAdd/ReadVariableOp.sequential_56/dense_122/BiasAdd/ReadVariableOp2^
-sequential_56/dense_122/MatMul/ReadVariableOp-sequential_56/dense_122/MatMul/ReadVariableOp2`
.sequential_56/dense_123/BiasAdd/ReadVariableOp.sequential_56/dense_123/BiasAdd/ReadVariableOp2^
-sequential_56/dense_123/MatMul/ReadVariableOp-sequential_56/dense_123/MatMul/ReadVariableOp:$ 

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

h
I__inference_dropout_66_layer_call_and_return_conditional_losses_266407662

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�?
�
L__inference_sequential_56_layer_call_and_return_conditional_losses_266408082

inputs
normalization_sub_y
normalization_sqrt_x:
(dense_121_matmul_readvariableop_resource: 7
)dense_121_biasadd_readvariableop_resource: :
(dense_122_matmul_readvariableop_resource:  7
)dense_122_biasadd_readvariableop_resource: :
(dense_123_matmul_readvariableop_resource: 7
)dense_123_biasadd_readvariableop_resource:
identity�� dense_121/BiasAdd/ReadVariableOp�dense_121/MatMul/ReadVariableOp�2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp� dense_122/BiasAdd/ReadVariableOp�dense_122/MatMul/ReadVariableOp�2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp� dense_123/BiasAdd/ReadVariableOp�dense_123/MatMul/ReadVariableOpg
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
dense_121/MatMul/ReadVariableOpReadVariableOp(dense_121_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_121/MatMulMatMulnormalization/truediv:z:0'dense_121/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
 dense_121/BiasAdd/ReadVariableOpReadVariableOp)dense_121_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_121/BiasAddBiasAdddense_121/MatMul:product:0(dense_121/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� d
dense_121/ReluReludense_121/BiasAdd:output:0*
T0*'
_output_shapes
:��������� ]
dropout_65/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?�
dropout_65/dropout/MulMuldense_121/Relu:activations:0!dropout_65/dropout/Const:output:0*
T0*'
_output_shapes
:��������� r
dropout_65/dropout/ShapeShapedense_121/Relu:activations:0*
T0*
_output_shapes
::���
/dropout_65/dropout/random_uniform/RandomUniformRandomUniform!dropout_65/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0f
!dropout_65/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout_65/dropout/GreaterEqualGreaterEqual8dropout_65/dropout/random_uniform/RandomUniform:output:0*dropout_65/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� _
dropout_65/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_65/dropout/SelectV2SelectV2#dropout_65/dropout/GreaterEqual:z:0dropout_65/dropout/Mul:z:0#dropout_65/dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� �
dense_122/MatMul/ReadVariableOpReadVariableOp(dense_122_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_122/MatMulMatMul$dropout_65/dropout/SelectV2:output:0'dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
 dense_122/BiasAdd/ReadVariableOpReadVariableOp)dense_122_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_122/BiasAddBiasAdddense_122/MatMul:product:0(dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� d
dense_122/ReluReludense_122/BiasAdd:output:0*
T0*'
_output_shapes
:��������� ]
dropout_66/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?�
dropout_66/dropout/MulMuldense_122/Relu:activations:0!dropout_66/dropout/Const:output:0*
T0*'
_output_shapes
:��������� r
dropout_66/dropout/ShapeShapedense_122/Relu:activations:0*
T0*
_output_shapes
::���
/dropout_66/dropout/random_uniform/RandomUniformRandomUniform!dropout_66/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0f
!dropout_66/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout_66/dropout/GreaterEqualGreaterEqual8dropout_66/dropout/random_uniform/RandomUniform:output:0*dropout_66/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� _
dropout_66/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_66/dropout/SelectV2SelectV2#dropout_66/dropout/GreaterEqual:z:0dropout_66/dropout/Mul:z:0#dropout_66/dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� �
dense_123/MatMul/ReadVariableOpReadVariableOp(dense_123_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
dense_123/MatMulMatMul$dropout_66/dropout/SelectV2:output:0'dense_123/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_123/BiasAdd/ReadVariableOpReadVariableOp)dense_123_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_123/BiasAddBiasAdddense_123/MatMul:product:0(dense_123/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2dense_121/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_121_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
#dense_121/kernel/Regularizer/L2LossL2Loss:dense_121/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_121/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
 dense_121/kernel/Regularizer/mulMul+dense_121/kernel/Regularizer/mul/x:output:0,dense_121/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_122/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_122_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
#dense_122/kernel/Regularizer/L2LossL2Loss:dense_122/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0,dense_122/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_123/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_121/BiasAdd/ReadVariableOp ^dense_121/MatMul/ReadVariableOp3^dense_121/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_122/BiasAdd/ReadVariableOp ^dense_122/MatMul/ReadVariableOp3^dense_122/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_123/BiasAdd/ReadVariableOp ^dense_123/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������::: : : : : : 2D
 dense_121/BiasAdd/ReadVariableOp dense_121/BiasAdd/ReadVariableOp2B
dense_121/MatMul/ReadVariableOpdense_121/MatMul/ReadVariableOp2h
2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_122/BiasAdd/ReadVariableOp dense_122/BiasAdd/ReadVariableOp2B
dense_122/MatMul/ReadVariableOpdense_122/MatMul/ReadVariableOp2h
2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_123/BiasAdd/ReadVariableOp dense_123/BiasAdd/ReadVariableOp2B
dense_123/MatMul/ReadVariableOpdense_123/MatMul/ReadVariableOp:$ 

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
H__inference_dense_121_layer_call_and_return_conditional_losses_266408147

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2dense_121/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� �
2dense_121/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0�
#dense_121/kernel/Regularizer/L2LossL2Loss:dense_121/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_121/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
 dense_121/kernel/Regularizer/mulMul+dense_121/kernel/Regularizer/mul/x:output:0,dense_121/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_121/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
1__inference_sequential_56_layer_call_fn_266407850
normalization_input
unknown
	unknown_0
	unknown_1: 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_56_layer_call_and_return_conditional_losses_266407831o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������::: : : : : : 22
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
1__inference_sequential_56_layer_call_fn_266408027

inputs
unknown
	unknown_0
	unknown_1: 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_56_layer_call_and_return_conditional_losses_266407831o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������::: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_output_shapes

::$ 

_output_shapes

::X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�)
�
L__inference_sequential_56_layer_call_and_return_conditional_losses_266407831

inputs
normalization_sub_y
normalization_sqrt_x%
dense_121_266407805: !
dense_121_266407807: %
dense_122_266407811:  !
dense_122_266407813: %
dense_123_266407817: !
dense_123_266407819:
identity��!dense_121/StatefulPartitionedCall�2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp�!dense_122/StatefulPartitionedCall�2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp�!dense_123/StatefulPartitionedCallg
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
!dense_121/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_121_266407805dense_121_266407807*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_121_layer_call_and_return_conditional_losses_266407609�
dropout_65/PartitionedCallPartitionedCall*dense_121/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dropout_65_layer_call_and_return_conditional_losses_266407708�
!dense_122/StatefulPartitionedCallStatefulPartitionedCall#dropout_65/PartitionedCall:output:0dense_122_266407811dense_122_266407813*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_122_layer_call_and_return_conditional_losses_266407644�
dropout_66/PartitionedCallPartitionedCall*dense_122/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dropout_66_layer_call_and_return_conditional_losses_266407719�
!dense_123/StatefulPartitionedCallStatefulPartitionedCall#dropout_66/PartitionedCall:output:0dense_123_266407817dense_123_266407819*
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
GPU 2J 8� *Q
fLRJ
H__inference_dense_123_layer_call_and_return_conditional_losses_266407674�
2dense_121/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_121_266407805*
_output_shapes

: *
dtype0�
#dense_121/kernel/Regularizer/L2LossL2Loss:dense_121/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_121/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
 dense_121/kernel/Regularizer/mulMul+dense_121/kernel/Regularizer/mul/x:output:0,dense_121/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_122/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_122_266407811*
_output_shapes

:  *
dtype0�
#dense_122/kernel/Regularizer/L2LossL2Loss:dense_122/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0,dense_122/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_123/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_121/StatefulPartitionedCall3^dense_121/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_122/StatefulPartitionedCall3^dense_122/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_123/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������::: : : : : : 2F
!dense_121/StatefulPartitionedCall!dense_121/StatefulPartitionedCall2h
2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2h
2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_123/StatefulPartitionedCall!dense_123/StatefulPartitionedCall:$ 

_output_shapes

::$ 

_output_shapes

::X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�k
�
%__inference__traced_restore_266408522
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:$
assignvariableop_2_count_1:	 5
#assignvariableop_3_dense_121_kernel: /
!assignvariableop_4_dense_121_bias: 5
#assignvariableop_5_dense_122_kernel:  /
!assignvariableop_6_dense_122_bias: 5
#assignvariableop_7_dense_123_kernel: /
!assignvariableop_8_dense_123_bias:&
assignvariableop_9_iteration:	 +
!assignvariableop_10_learning_rate: =
+assignvariableop_11_adam_m_dense_121_kernel: =
+assignvariableop_12_adam_v_dense_121_kernel: 7
)assignvariableop_13_adam_m_dense_121_bias: 7
)assignvariableop_14_adam_v_dense_121_bias: =
+assignvariableop_15_adam_m_dense_122_kernel:  =
+assignvariableop_16_adam_v_dense_122_kernel:  7
)assignvariableop_17_adam_m_dense_122_bias: 7
)assignvariableop_18_adam_v_dense_122_bias: =
+assignvariableop_19_adam_m_dense_123_kernel: =
+assignvariableop_20_adam_v_dense_123_kernel: 7
)assignvariableop_21_adam_m_dense_123_bias:7
)assignvariableop_22_adam_v_dense_123_bias:#
assignvariableop_23_total: #
assignvariableop_24_count: 
identity_26��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2		[
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
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_121_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_121_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_122_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_122_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_123_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_123_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_iterationIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp!assignvariableop_10_learning_rateIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp+assignvariableop_11_adam_m_dense_121_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp+assignvariableop_12_adam_v_dense_121_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_m_dense_121_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_v_dense_121_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_m_dense_122_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp+assignvariableop_16_adam_v_dense_122_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_m_dense_122_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_v_dense_122_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_m_dense_123_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp+assignvariableop_20_adam_v_dense_123_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_m_dense_123_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_v_dense_123_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_26IdentityIdentity_25:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_26Identity_26:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_2AssignVariableOp_22(
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
�
�
-__inference_dense_123_layer_call_fn_266408234

inputs
unknown: 
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
GPU 2J 8� *Q
fLRJ
H__inference_dense_123_layer_call_and_return_conditional_losses_266407674o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�*
�
L__inference_sequential_56_layer_call_and_return_conditional_losses_266407735
normalization_input
normalization_sub_y
normalization_sqrt_x%
dense_121_266407699: !
dense_121_266407701: %
dense_122_266407710:  !
dense_122_266407712: %
dense_123_266407721: !
dense_123_266407723:
identity��!dense_121/StatefulPartitionedCall�2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp�!dense_122/StatefulPartitionedCall�2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp�!dense_123/StatefulPartitionedCallt
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
!dense_121/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_121_266407699dense_121_266407701*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_121_layer_call_and_return_conditional_losses_266407609�
dropout_65/PartitionedCallPartitionedCall*dense_121/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dropout_65_layer_call_and_return_conditional_losses_266407708�
!dense_122/StatefulPartitionedCallStatefulPartitionedCall#dropout_65/PartitionedCall:output:0dense_122_266407710dense_122_266407712*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_122_layer_call_and_return_conditional_losses_266407644�
dropout_66/PartitionedCallPartitionedCall*dense_122/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dropout_66_layer_call_and_return_conditional_losses_266407719�
!dense_123/StatefulPartitionedCallStatefulPartitionedCall#dropout_66/PartitionedCall:output:0dense_123_266407721dense_123_266407723*
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
GPU 2J 8� *Q
fLRJ
H__inference_dense_123_layer_call_and_return_conditional_losses_266407674�
2dense_121/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_121_266407699*
_output_shapes

: *
dtype0�
#dense_121/kernel/Regularizer/L2LossL2Loss:dense_121/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_121/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
 dense_121/kernel/Regularizer/mulMul+dense_121/kernel/Regularizer/mul/x:output:0,dense_121/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_122/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_122_266407710*
_output_shapes

:  *
dtype0�
#dense_122/kernel/Regularizer/L2LossL2Loss:dense_122/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0,dense_122/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_123/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_121/StatefulPartitionedCall3^dense_121/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_122/StatefulPartitionedCall3^dense_122/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_123/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������::: : : : : : 2F
!dense_121/StatefulPartitionedCall!dense_121/StatefulPartitionedCall2h
2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2h
2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_123/StatefulPartitionedCall!dense_123/StatefulPartitionedCall:$ 

_output_shapes

::$ 

_output_shapes

::e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�
�
H__inference_dense_121_layer_call_and_return_conditional_losses_266407609

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�2dense_121/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� �
2dense_121/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0�
#dense_121/kernel/Regularizer/L2LossL2Loss:dense_121/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_121/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
 dense_121/kernel/Regularizer/mulMul+dense_121/kernel/Regularizer/mul/x:output:0,dense_121/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_121/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
J
.__inference_dropout_66_layer_call_fn_266408208

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dropout_66_layer_call_and_return_conditional_losses_266407719`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
g
I__inference_dropout_66_layer_call_and_return_conditional_losses_266408225

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
g
I__inference_dropout_65_layer_call_and_return_conditional_losses_266407708

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
'__inference_signature_wrapper_266407977
normalization_input
unknown
	unknown_0
	unknown_1: 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__wrapped_model_266407583o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������::: : : : : : 22
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
�
g
.__inference_dropout_66_layer_call_fn_266408203

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dropout_66_layer_call_and_return_conditional_losses_266407662o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

h
I__inference_dropout_65_layer_call_and_return_conditional_losses_266407627

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
1__inference_sequential_56_layer_call_fn_266407793
normalization_input
unknown
	unknown_0
	unknown_1: 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_56_layer_call_and_return_conditional_losses_266407774o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������::: : : : : : 22
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
�
g
I__inference_dropout_65_layer_call_and_return_conditional_losses_266408174

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
̸
�
"__inference__traced_save_266408437
file_prefix)
read_disablecopyonread_mean:/
!read_1_disablecopyonread_variance:*
 read_2_disablecopyonread_count_1:	 ;
)read_3_disablecopyonread_dense_121_kernel: 5
'read_4_disablecopyonread_dense_121_bias: ;
)read_5_disablecopyonread_dense_122_kernel:  5
'read_6_disablecopyonread_dense_122_bias: ;
)read_7_disablecopyonread_dense_123_kernel: 5
'read_8_disablecopyonread_dense_123_bias:,
"read_9_disablecopyonread_iteration:	 1
'read_10_disablecopyonread_learning_rate: C
1read_11_disablecopyonread_adam_m_dense_121_kernel: C
1read_12_disablecopyonread_adam_v_dense_121_kernel: =
/read_13_disablecopyonread_adam_m_dense_121_bias: =
/read_14_disablecopyonread_adam_v_dense_121_bias: C
1read_15_disablecopyonread_adam_m_dense_122_kernel:  C
1read_16_disablecopyonread_adam_v_dense_122_kernel:  =
/read_17_disablecopyonread_adam_m_dense_122_bias: =
/read_18_disablecopyonread_adam_v_dense_122_bias: C
1read_19_disablecopyonread_adam_m_dense_123_kernel: C
1read_20_disablecopyonread_adam_v_dense_123_kernel: =
/read_21_disablecopyonread_adam_m_dense_123_bias:=
/read_22_disablecopyonread_adam_v_dense_123_bias:)
read_23_disablecopyonread_total: )
read_24_disablecopyonread_count: 
savev2_const_2
identity_51��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
: }
Read_3/DisableCopyOnReadDisableCopyOnRead)read_3_disablecopyonread_dense_121_kernel"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp)read_3_disablecopyonread_dense_121_kernel^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0m

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: c

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes

: {
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_dense_121_bias"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_dense_121_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_5/DisableCopyOnReadDisableCopyOnRead)read_5_disablecopyonread_dense_122_kernel"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp)read_5_disablecopyonread_dense_122_kernel^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0n
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes

:  {
Read_6/DisableCopyOnReadDisableCopyOnRead'read_6_disablecopyonread_dense_122_bias"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp'read_6_disablecopyonread_dense_122_bias^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_7/DisableCopyOnReadDisableCopyOnRead)read_7_disablecopyonread_dense_123_kernel"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp)read_7_disablecopyonread_dense_123_kernel^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0n
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes

: {
Read_8/DisableCopyOnReadDisableCopyOnRead'read_8_disablecopyonread_dense_123_bias"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp'read_8_disablecopyonread_dense_123_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_9/DisableCopyOnReadDisableCopyOnRead"read_9_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp"read_9_disablecopyonread_iteration^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_10/DisableCopyOnReadDisableCopyOnRead'read_10_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp'read_10_disablecopyonread_learning_rate^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_11/DisableCopyOnReadDisableCopyOnRead1read_11_disablecopyonread_adam_m_dense_121_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp1read_11_disablecopyonread_adam_m_dense_121_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_12/DisableCopyOnReadDisableCopyOnRead1read_12_disablecopyonread_adam_v_dense_121_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp1read_12_disablecopyonread_adam_v_dense_121_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_13/DisableCopyOnReadDisableCopyOnRead/read_13_disablecopyonread_adam_m_dense_121_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp/read_13_disablecopyonread_adam_m_dense_121_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_14/DisableCopyOnReadDisableCopyOnRead/read_14_disablecopyonread_adam_v_dense_121_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp/read_14_disablecopyonread_adam_v_dense_121_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_15/DisableCopyOnReadDisableCopyOnRead1read_15_disablecopyonread_adam_m_dense_122_kernel"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp1read_15_disablecopyonread_adam_m_dense_122_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_16/DisableCopyOnReadDisableCopyOnRead1read_16_disablecopyonread_adam_v_dense_122_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp1read_16_disablecopyonread_adam_v_dense_122_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_17/DisableCopyOnReadDisableCopyOnRead/read_17_disablecopyonread_adam_m_dense_122_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp/read_17_disablecopyonread_adam_m_dense_122_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_18/DisableCopyOnReadDisableCopyOnRead/read_18_disablecopyonread_adam_v_dense_122_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp/read_18_disablecopyonread_adam_v_dense_122_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_19/DisableCopyOnReadDisableCopyOnRead1read_19_disablecopyonread_adam_m_dense_123_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp1read_19_disablecopyonread_adam_m_dense_123_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_20/DisableCopyOnReadDisableCopyOnRead1read_20_disablecopyonread_adam_v_dense_123_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp1read_20_disablecopyonread_adam_v_dense_123_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_21/DisableCopyOnReadDisableCopyOnRead/read_21_disablecopyonread_adam_m_dense_123_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp/read_21_disablecopyonread_adam_m_dense_123_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_22/DisableCopyOnReadDisableCopyOnRead/read_22_disablecopyonread_adam_v_dense_123_bias"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp/read_22_disablecopyonread_adam_v_dense_123_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_23/DisableCopyOnReadDisableCopyOnReadread_23_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOpread_23_disablecopyonread_total^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_24/DisableCopyOnReadDisableCopyOnReadread_24_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOpread_24_disablecopyonread_count^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0savev2_const_2"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *(
dtypes
2		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_50Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_51IdentityIdentity_50:output:0^NoOp*
T0*
_output_shapes
: �

NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_51Identity_51:output:0*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_24/ReadVariableOpRead_24/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
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
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
-__inference_dense_121_layer_call_fn_266408132

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_121_layer_call_and_return_conditional_losses_266407609o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
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
�,
�
L__inference_sequential_56_layer_call_and_return_conditional_losses_266407774

inputs
normalization_sub_y
normalization_sqrt_x%
dense_121_266407748: !
dense_121_266407750: %
dense_122_266407754:  !
dense_122_266407756: %
dense_123_266407760: !
dense_123_266407762:
identity��!dense_121/StatefulPartitionedCall�2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp�!dense_122/StatefulPartitionedCall�2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp�!dense_123/StatefulPartitionedCall�"dropout_65/StatefulPartitionedCall�"dropout_66/StatefulPartitionedCallg
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
!dense_121/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_121_266407748dense_121_266407750*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_121_layer_call_and_return_conditional_losses_266407609�
"dropout_65/StatefulPartitionedCallStatefulPartitionedCall*dense_121/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dropout_65_layer_call_and_return_conditional_losses_266407627�
!dense_122/StatefulPartitionedCallStatefulPartitionedCall+dropout_65/StatefulPartitionedCall:output:0dense_122_266407754dense_122_266407756*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_122_layer_call_and_return_conditional_losses_266407644�
"dropout_66/StatefulPartitionedCallStatefulPartitionedCall*dense_122/StatefulPartitionedCall:output:0#^dropout_65/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_dropout_66_layer_call_and_return_conditional_losses_266407662�
!dense_123/StatefulPartitionedCallStatefulPartitionedCall+dropout_66/StatefulPartitionedCall:output:0dense_123_266407760dense_123_266407762*
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
GPU 2J 8� *Q
fLRJ
H__inference_dense_123_layer_call_and_return_conditional_losses_266407674�
2dense_121/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_121_266407748*
_output_shapes

: *
dtype0�
#dense_121/kernel/Regularizer/L2LossL2Loss:dense_121/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_121/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
 dense_121/kernel/Regularizer/mulMul+dense_121/kernel/Regularizer/mul/x:output:0,dense_121/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
2dense_122/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_122_266407754*
_output_shapes

:  *
dtype0�
#dense_122/kernel/Regularizer/L2LossL2Loss:dense_122/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�Q9�
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0,dense_122/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_123/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_121/StatefulPartitionedCall3^dense_121/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_122/StatefulPartitionedCall3^dense_122/kernel/Regularizer/L2Loss/ReadVariableOp"^dense_123/StatefulPartitionedCall#^dropout_65/StatefulPartitionedCall#^dropout_66/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:������������������::: : : : : : 2F
!dense_121/StatefulPartitionedCall!dense_121/StatefulPartitionedCall2h
2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp2dense_121/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2h
2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp2dense_122/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dense_123/StatefulPartitionedCall!dense_123/StatefulPartitionedCall2H
"dropout_65/StatefulPartitionedCall"dropout_65/StatefulPartitionedCall2H
"dropout_66/StatefulPartitionedCall"dropout_66/StatefulPartitionedCall:$ 

_output_shapes

::$ 

_output_shapes

::X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�	
�
H__inference_dense_123_layer_call_and_return_conditional_losses_266407674

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
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
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
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
%serving_default_normalization_input:0������������������=
	dense_1230
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
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
_adapt_function"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias"
_tf_keras_layer
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_random_generator"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_random_generator"
_tf_keras_layer
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias"
_tf_keras_layer
_
0
1
2
3
 4
.5
/6
=7
>8"
trackable_list_wrapper
J
0
 1
.2
/3
=4
>5"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
�
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ftrace_0
Gtrace_1
Htrace_2
Itrace_32�
1__inference_sequential_56_layer_call_fn_266407793
1__inference_sequential_56_layer_call_fn_266407850
1__inference_sequential_56_layer_call_fn_266408006
1__inference_sequential_56_layer_call_fn_266408027�
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
 zFtrace_0zGtrace_1zHtrace_2zItrace_3
�
Jtrace_0
Ktrace_1
Ltrace_2
Mtrace_32�
L__inference_sequential_56_layer_call_and_return_conditional_losses_266407689
L__inference_sequential_56_layer_call_and_return_conditional_losses_266407735
L__inference_sequential_56_layer_call_and_return_conditional_losses_266408082
L__inference_sequential_56_layer_call_and_return_conditional_losses_266408123�
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
 zJtrace_0zKtrace_1zLtrace_2zMtrace_3
�
N	capture_0
O	capture_1B�
$__inference__wrapped_model_266407583normalization_input"�
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
 zN	capture_0zO	capture_1
�
P
_variables
Q_iterations
R_learning_rate
S_index_dict
T
_momentums
U_velocities
V_update_step_xla"
experimentalOptimizer
,
Wserving_default"
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
Xtrace_02�
__inference_adapt_step_2715533�
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
 zXtrace_0
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
'
?0"
trackable_list_wrapper
�
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
^trace_02�
-__inference_dense_121_layer_call_fn_266408132�
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
 z^trace_0
�
_trace_02�
H__inference_dense_121_layer_call_and_return_conditional_losses_266408147�
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
 z_trace_0
":  2dense_121/kernel
: 2dense_121/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�
etrace_0
ftrace_12�
.__inference_dropout_65_layer_call_fn_266408152
.__inference_dropout_65_layer_call_fn_266408157�
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
 zetrace_0zftrace_1
�
gtrace_0
htrace_12�
I__inference_dropout_65_layer_call_and_return_conditional_losses_266408169
I__inference_dropout_65_layer_call_and_return_conditional_losses_266408174�
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
 zgtrace_0zhtrace_1
"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
'
@0"
trackable_list_wrapper
�
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
ntrace_02�
-__inference_dense_122_layer_call_fn_266408183�
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
 zntrace_0
�
otrace_02�
H__inference_dense_122_layer_call_and_return_conditional_losses_266408198�
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
 zotrace_0
":   2dense_122/kernel
: 2dense_122/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
utrace_0
vtrace_12�
.__inference_dropout_66_layer_call_fn_266408203
.__inference_dropout_66_layer_call_fn_266408208�
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
 zutrace_0zvtrace_1
�
wtrace_0
xtrace_12�
I__inference_dropout_66_layer_call_and_return_conditional_losses_266408220
I__inference_dropout_66_layer_call_and_return_conditional_losses_266408225�
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
"
_generic_user_object
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�
~trace_02�
-__inference_dense_123_layer_call_fn_266408234�
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
 z~trace_0
�
trace_02�
H__inference_dense_123_layer_call_and_return_conditional_losses_266408244�
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
 ztrace_0
":  2dense_123/kernel
:2dense_123/bias
�
�trace_02�
__inference_loss_fn_0_266408253�
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
__inference_loss_fn_1_266408262�
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
0
1
2"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
N	capture_0
O	capture_1B�
1__inference_sequential_56_layer_call_fn_266407793normalization_input"�
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
 zN	capture_0zO	capture_1
�
N	capture_0
O	capture_1B�
1__inference_sequential_56_layer_call_fn_266407850normalization_input"�
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
 zN	capture_0zO	capture_1
�
N	capture_0
O	capture_1B�
1__inference_sequential_56_layer_call_fn_266408006inputs"�
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
 zN	capture_0zO	capture_1
�
N	capture_0
O	capture_1B�
1__inference_sequential_56_layer_call_fn_266408027inputs"�
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
 zN	capture_0zO	capture_1
�
N	capture_0
O	capture_1B�
L__inference_sequential_56_layer_call_and_return_conditional_losses_266407689normalization_input"�
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
 zN	capture_0zO	capture_1
�
N	capture_0
O	capture_1B�
L__inference_sequential_56_layer_call_and_return_conditional_losses_266407735normalization_input"�
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
 zN	capture_0zO	capture_1
�
N	capture_0
O	capture_1B�
L__inference_sequential_56_layer_call_and_return_conditional_losses_266408082inputs"�
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
 zN	capture_0zO	capture_1
�
N	capture_0
O	capture_1B�
L__inference_sequential_56_layer_call_and_return_conditional_losses_266408123inputs"�
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
 zN	capture_0zO	capture_1
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
�
Q0
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
�12"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
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
N	capture_0
O	capture_1B�
'__inference_signature_wrapper_266407977normalization_input"�
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
 zN	capture_0zO	capture_1
�B�
__inference_adapt_step_2715533iterator"�
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
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_dense_121_layer_call_fn_266408132inputs"�
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
H__inference_dense_121_layer_call_and_return_conditional_losses_266408147inputs"�
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
.__inference_dropout_65_layer_call_fn_266408152inputs"�
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
.__inference_dropout_65_layer_call_fn_266408157inputs"�
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
I__inference_dropout_65_layer_call_and_return_conditional_losses_266408169inputs"�
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
I__inference_dropout_65_layer_call_and_return_conditional_losses_266408174inputs"�
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
@0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_dense_122_layer_call_fn_266408183inputs"�
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
H__inference_dense_122_layer_call_and_return_conditional_losses_266408198inputs"�
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
.__inference_dropout_66_layer_call_fn_266408203inputs"�
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
.__inference_dropout_66_layer_call_fn_266408208inputs"�
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
I__inference_dropout_66_layer_call_and_return_conditional_losses_266408220inputs"�
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
I__inference_dropout_66_layer_call_and_return_conditional_losses_266408225inputs"�
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
-__inference_dense_123_layer_call_fn_266408234inputs"�
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
H__inference_dense_123_layer_call_and_return_conditional_losses_266408244inputs"�
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
__inference_loss_fn_0_266408253"�
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
__inference_loss_fn_1_266408262"�
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
':% 2Adam/m/dense_121/kernel
':% 2Adam/v/dense_121/kernel
!: 2Adam/m/dense_121/bias
!: 2Adam/v/dense_121/bias
':%  2Adam/m/dense_122/kernel
':%  2Adam/v/dense_122/kernel
!: 2Adam/m/dense_122/bias
!: 2Adam/v/dense_122/bias
':% 2Adam/m/dense_123/kernel
':% 2Adam/v/dense_123/kernel
!:2Adam/m/dense_123/bias
!:2Adam/v/dense_123/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count�
$__inference__wrapped_model_266407583�NO ./=>E�B
;�8
6�3
normalization_input������������������
� "5�2
0
	dense_123#� 
	dense_123���������g
__inference_adapt_step_2715533E:�7
0�-
+�(�
� IteratorSpec 
� "
 �
H__inference_dense_121_layer_call_and_return_conditional_losses_266408147c /�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0��������� 
� �
-__inference_dense_121_layer_call_fn_266408132X /�,
%�"
 �
inputs���������
� "!�
unknown��������� �
H__inference_dense_122_layer_call_and_return_conditional_losses_266408198c.//�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� �
-__inference_dense_122_layer_call_fn_266408183X.//�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
H__inference_dense_123_layer_call_and_return_conditional_losses_266408244c=>/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������
� �
-__inference_dense_123_layer_call_fn_266408234X=>/�,
%�"
 �
inputs��������� 
� "!�
unknown����������
I__inference_dropout_65_layer_call_and_return_conditional_losses_266408169c3�0
)�&
 �
inputs��������� 
p
� ",�)
"�
tensor_0��������� 
� �
I__inference_dropout_65_layer_call_and_return_conditional_losses_266408174c3�0
)�&
 �
inputs��������� 
p 
� ",�)
"�
tensor_0��������� 
� �
.__inference_dropout_65_layer_call_fn_266408152X3�0
)�&
 �
inputs��������� 
p
� "!�
unknown��������� �
.__inference_dropout_65_layer_call_fn_266408157X3�0
)�&
 �
inputs��������� 
p 
� "!�
unknown��������� �
I__inference_dropout_66_layer_call_and_return_conditional_losses_266408220c3�0
)�&
 �
inputs��������� 
p
� ",�)
"�
tensor_0��������� 
� �
I__inference_dropout_66_layer_call_and_return_conditional_losses_266408225c3�0
)�&
 �
inputs��������� 
p 
� ",�)
"�
tensor_0��������� 
� �
.__inference_dropout_66_layer_call_fn_266408203X3�0
)�&
 �
inputs��������� 
p
� "!�
unknown��������� �
.__inference_dropout_66_layer_call_fn_266408208X3�0
)�&
 �
inputs��������� 
p 
� "!�
unknown��������� G
__inference_loss_fn_0_266408253$�

� 
� "�
unknown G
__inference_loss_fn_1_266408262$.�

� 
� "�
unknown �
L__inference_sequential_56_layer_call_and_return_conditional_losses_266407689�NO ./=>M�J
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
L__inference_sequential_56_layer_call_and_return_conditional_losses_266407735�NO ./=>M�J
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
L__inference_sequential_56_layer_call_and_return_conditional_losses_266408082zNO ./=>@�=
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
L__inference_sequential_56_layer_call_and_return_conditional_losses_266408123zNO ./=>@�=
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
1__inference_sequential_56_layer_call_fn_266407793|NO ./=>M�J
C�@
6�3
normalization_input������������������
p

 
� "!�
unknown����������
1__inference_sequential_56_layer_call_fn_266407850|NO ./=>M�J
C�@
6�3
normalization_input������������������
p 

 
� "!�
unknown����������
1__inference_sequential_56_layer_call_fn_266408006oNO ./=>@�=
6�3
)�&
inputs������������������
p

 
� "!�
unknown����������
1__inference_sequential_56_layer_call_fn_266408027oNO ./=>@�=
6�3
)�&
inputs������������������
p 

 
� "!�
unknown����������
'__inference_signature_wrapper_266407977�NO ./=>\�Y
� 
R�O
M
normalization_input6�3
normalization_input������������������"5�2
0
	dense_123#� 
	dense_123���������