▓╦
Л▄
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
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
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
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
о
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
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
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
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
┴
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.9.02v2.9.0-rc2-42-g8a20d54a3c18Ми
И
RMSprop/dense_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_1/bias/rms
Б
,RMSprop/dense_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/bias/rms*
_output_shapes
:*
dtype0
Р
RMSprop/dense_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*+
shared_nameRMSprop/dense_1/kernel/rms
Й
.RMSprop/dense_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/kernel/rms*
_output_shapes

:
*
dtype0
Т
RMSprop/conv1d/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameRMSprop/conv1d/kernel/rms
Л
-RMSprop/conv1d/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1d/kernel/rms*"
_output_shapes
:*
dtype0
─
4RMSprop/multi_level__block_attention/output_bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*E
shared_name64RMSprop/multi_level__block_attention/output_bias/rms
╜
HRMSprop/multi_level__block_attention/output_bias/rms/Read/ReadVariableOpReadVariableOp4RMSprop/multi_level__block_attention/output_bias/rms*
_output_shapes

:
*
dtype0
┬
3RMSprop/multi_level__block_attention/value_bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*D
shared_name53RMSprop/multi_level__block_attention/value_bias/rms
╗
GRMSprop/multi_level__block_attention/value_bias/rms/Read/ReadVariableOpReadVariableOp3RMSprop/multi_level__block_attention/value_bias/rms*
_output_shapes

:
*
dtype0
╛
1RMSprop/multi_level__block_attention/key_bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*B
shared_name31RMSprop/multi_level__block_attention/key_bias/rms
╖
ERMSprop/multi_level__block_attention/key_bias/rms/Read/ReadVariableOpReadVariableOp1RMSprop/multi_level__block_attention/key_bias/rms*
_output_shapes

:
*
dtype0
┬
3RMSprop/multi_level__block_attention/query_bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*D
shared_name53RMSprop/multi_level__block_attention/query_bias/rms
╗
GRMSprop/multi_level__block_attention/query_bias/rms/Read/ReadVariableOpReadVariableOp3RMSprop/multi_level__block_attention/query_bias/rms*
_output_shapes

:
*
dtype0
ф
DRMSprop/multi_level__block_attention/Epigenome_embedding_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*U
shared_nameFDRMSprop/multi_level__block_attention/Epigenome_embedding_weights/rms
▌
XRMSprop/multi_level__block_attention/Epigenome_embedding_weights/rms/Read/ReadVariableOpReadVariableOpDRMSprop/multi_level__block_attention/Epigenome_embedding_weights/rms*
_output_shapes

:

*
dtype0
ъ
GRMSprop/multi_level__block_attention/Annotation_embedding_v_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*X
shared_nameIGRMSprop/multi_level__block_attention/Annotation_embedding_v_weights/rms
у
[RMSprop/multi_level__block_attention/Annotation_embedding_v_weights/rms/Read/ReadVariableOpReadVariableOpGRMSprop/multi_level__block_attention/Annotation_embedding_v_weights/rms*
_output_shapes

:*
dtype0
ъ
GRMSprop/multi_level__block_attention/Annotation_embedding_k_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*X
shared_nameIGRMSprop/multi_level__block_attention/Annotation_embedding_k_weights/rms
у
[RMSprop/multi_level__block_attention/Annotation_embedding_k_weights/rms/Read/ReadVariableOpReadVariableOpGRMSprop/multi_level__block_attention/Annotation_embedding_k_weights/rms*
_output_shapes

:*
dtype0
ъ
GRMSprop/multi_level__block_attention/Annotation_embedding_q_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*X
shared_nameIGRMSprop/multi_level__block_attention/Annotation_embedding_q_weights/rms
у
[RMSprop/multi_level__block_attention/Annotation_embedding_q_weights/rms/Read/ReadVariableOpReadVariableOpGRMSprop/multi_level__block_attention/Annotation_embedding_q_weights/rms*
_output_shapes

:*
dtype0
Д
RMSprop/dense/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameRMSprop/dense/bias/rms
}
*RMSprop/dense/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/bias/rms*
_output_shapes
:*
dtype0
М
RMSprop/dense/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameRMSprop/dense/kernel/rms
Е
,RMSprop/dense/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/kernel/rms*
_output_shapes

:*
dtype0
м
&RMSprop/locally_connected1d/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*7
shared_name(&RMSprop/locally_connected1d/kernel/rms
е
:RMSprop/locally_connected1d/kernel/rms/Read/ReadVariableOpReadVariableOp&RMSprop/locally_connected1d/kernel/rms*"
_output_shapes
:

*
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
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
Z
rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namerho
S
rho/Read/ReadVariableOpReadVariableOprho*
_output_shapes
: *
dtype0
d
momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
momentum
]
momentum/Read/ReadVariableOpReadVariableOpmomentum*
_output_shapes
: *
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
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:
*
dtype0
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:*
dtype0
м
(multi_level__block_attention/output_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*9
shared_name*(multi_level__block_attention/output_bias
е
<multi_level__block_attention/output_bias/Read/ReadVariableOpReadVariableOp(multi_level__block_attention/output_bias*
_output_shapes

:
*
dtype0
к
'multi_level__block_attention/value_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*8
shared_name)'multi_level__block_attention/value_bias
г
;multi_level__block_attention/value_bias/Read/ReadVariableOpReadVariableOp'multi_level__block_attention/value_bias*
_output_shapes

:
*
dtype0
ж
%multi_level__block_attention/key_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*6
shared_name'%multi_level__block_attention/key_bias
Я
9multi_level__block_attention/key_bias/Read/ReadVariableOpReadVariableOp%multi_level__block_attention/key_bias*
_output_shapes

:
*
dtype0
к
'multi_level__block_attention/query_biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*8
shared_name)'multi_level__block_attention/query_bias
г
;multi_level__block_attention/query_bias/Read/ReadVariableOpReadVariableOp'multi_level__block_attention/query_bias*
_output_shapes

:
*
dtype0
╠
8multi_level__block_attention/Epigenome_embedding_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*I
shared_name:8multi_level__block_attention/Epigenome_embedding_weights
┼
Lmulti_level__block_attention/Epigenome_embedding_weights/Read/ReadVariableOpReadVariableOp8multi_level__block_attention/Epigenome_embedding_weights*
_output_shapes

:

*
dtype0
╥
;multi_level__block_attention/Annotation_embedding_v_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*L
shared_name=;multi_level__block_attention/Annotation_embedding_v_weights
╦
Omulti_level__block_attention/Annotation_embedding_v_weights/Read/ReadVariableOpReadVariableOp;multi_level__block_attention/Annotation_embedding_v_weights*
_output_shapes

:*
dtype0
╥
;multi_level__block_attention/Annotation_embedding_k_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*L
shared_name=;multi_level__block_attention/Annotation_embedding_k_weights
╦
Omulti_level__block_attention/Annotation_embedding_k_weights/Read/ReadVariableOpReadVariableOp;multi_level__block_attention/Annotation_embedding_k_weights*
_output_shapes

:*
dtype0
╥
;multi_level__block_attention/Annotation_embedding_q_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*L
shared_name=;multi_level__block_attention/Annotation_embedding_q_weights
╦
Omulti_level__block_attention/Annotation_embedding_q_weights/Read/ReadVariableOpReadVariableOp;multi_level__block_attention/Annotation_embedding_q_weights*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
Ф
locally_connected1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*+
shared_namelocally_connected1d/kernel
Н
.locally_connected1d/kernel/Read/ReadVariableOpReadVariableOplocally_connected1d/kernel*"
_output_shapes
:

*
dtype0

NoOpNoOp
│d
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*юc
valueфcBсc B┌c
ь
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
О
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
Ь
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel*
ж
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias*
н
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
"1Annotation_embedding_q_weights
	1Wq_ld
"2Annotation_embedding_k_weights
	2Wk_ld
"3Annotation_embedding_v_weights
	3Wv_ld
4Epigenome_embedding_weights
4
Wepigenome
5
query_bias
5bq
6key_bias
6bk
7
value_bias
7bv
8output_bias
8bo*
╛
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
 @_jit_compiled_convolution_op*
О
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses* 
О
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses* 
О
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses* 
О
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses* 
ж
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses

_kernel
`bias*
О
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses* 
j
"0
)1
*2
13
24
35
46
57
68
79
810
?11
_12
`13*
j
"0
)1
*2
13
24
35
46
57
68
79
810
?11
_12
`13*
* 
░
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
ltrace_0
mtrace_1
ntrace_2
otrace_3* 
6
ptrace_0
qtrace_1
rtrace_2
strace_3* 
* 
ы
	tdecay
ulearning_rate
vmomentum
wrho
xiter
"rms╥
)rms╙
*rms╘
1rms╒
2rms╓
3rms╫
4rms╪
5rms┘
6rms┌
7rms█
8rms▄
?rms▌
_rms▐
`rms▀*

yserving_default* 
* 
* 
* 
С
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

trace_0* 

Аtrace_0* 

"0*

"0*
* 
Ш
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

Жtrace_0* 

Зtrace_0* 
jd
VARIABLE_VALUElocally_connected1d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*

)0
*1*

)0
*1*
* 
Ш
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

Нtrace_0* 

Оtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
<
10
21
32
43
54
65
76
87*
<
10
21
32
43
54
65
76
87*
* 
Ш
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*

Фtrace_0* 

Хtrace_0* 
дЭ
VARIABLE_VALUE;multi_level__block_attention/Annotation_embedding_q_weightsNlayer_with_weights-2/Annotation_embedding_q_weights/.ATTRIBUTES/VARIABLE_VALUE*
дЭ
VARIABLE_VALUE;multi_level__block_attention/Annotation_embedding_k_weightsNlayer_with_weights-2/Annotation_embedding_k_weights/.ATTRIBUTES/VARIABLE_VALUE*
дЭ
VARIABLE_VALUE;multi_level__block_attention/Annotation_embedding_v_weightsNlayer_with_weights-2/Annotation_embedding_v_weights/.ATTRIBUTES/VARIABLE_VALUE*
ЮЧ
VARIABLE_VALUE8multi_level__block_attention/Epigenome_embedding_weightsKlayer_with_weights-2/Epigenome_embedding_weights/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE'multi_level__block_attention/query_bias:layer_with_weights-2/query_bias/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE%multi_level__block_attention/key_bias8layer_with_weights-2/key_bias/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE'multi_level__block_attention/value_bias:layer_with_weights-2/value_bias/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE(multi_level__block_attention/output_bias;layer_with_weights-2/output_bias/.ATTRIBUTES/VARIABLE_VALUE*

?0*

?0*
* 
Ш
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

Ыtrace_0* 

Ьtrace_0* 
]W
VARIABLE_VALUEconv1d/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses* 

вtrace_0* 

гtrace_0* 
* 
* 
* 
Ц
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 

йtrace_0* 

кtrace_0* 
* 
* 
* 
Ц
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 

░trace_0* 

▒trace_0* 
* 
* 
* 
Ц
▓non_trainable_variables
│layers
┤metrics
 ╡layer_regularization_losses
╢layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses* 

╖trace_0* 

╕trace_0* 

_0
`1*

_0
`1*
* 
Ш
╣non_trainable_variables
║layers
╗metrics
 ╝layer_regularization_losses
╜layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses*

╛trace_0* 

┐trace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
└non_trainable_variables
┴layers
┬metrics
 ├layer_regularization_losses
─layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses* 

┼trace_0* 

╞trace_0* 
* 
Z
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
11*

╟0
╚1*
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
IC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
E?
VARIABLE_VALUErho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
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
╔	variables
╩	keras_api

╦total

╠count*
M
═	variables
╬	keras_api

╧total

╨count
╤
_fn_kwargs*

╦0
╠1*

╔	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

╧0
╨1*

═	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
ХО
VARIABLE_VALUE&RMSprop/locally_connected1d/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUERMSprop/dense/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUERMSprop/dense/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
╬╟
VARIABLE_VALUEGRMSprop/multi_level__block_attention/Annotation_embedding_q_weights/rmsllayer_with_weights-2/Annotation_embedding_q_weights/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
╬╟
VARIABLE_VALUEGRMSprop/multi_level__block_attention/Annotation_embedding_k_weights/rmsllayer_with_weights-2/Annotation_embedding_k_weights/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
╬╟
VARIABLE_VALUEGRMSprop/multi_level__block_attention/Annotation_embedding_v_weights/rmsllayer_with_weights-2/Annotation_embedding_v_weights/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
╚┴
VARIABLE_VALUEDRMSprop/multi_level__block_attention/Epigenome_embedding_weights/rmsilayer_with_weights-2/Epigenome_embedding_weights/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
жЯ
VARIABLE_VALUE3RMSprop/multi_level__block_attention/query_bias/rmsXlayer_with_weights-2/query_bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
вЫ
VARIABLE_VALUE1RMSprop/multi_level__block_attention/key_bias/rmsVlayer_with_weights-2/key_bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
жЯ
VARIABLE_VALUE3RMSprop/multi_level__block_attention/value_bias/rmsXlayer_with_weights-2/value_bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
иб
VARIABLE_VALUE4RMSprop/multi_level__block_attention/output_bias/rmsYlayer_with_weights-2/output_bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
ИБ
VARIABLE_VALUERMSprop/conv1d/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUERMSprop/dense_1/kernel/rmsTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUERMSprop/dense_1/bias/rmsRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
И
serving_default_input_layer_1Placeholder*+
_output_shapes
:         d*
dtype0* 
shape:         d
╦
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layer_1locally_connected1d/kerneldense/kernel
dense/bias;multi_level__block_attention/Annotation_embedding_q_weights;multi_level__block_attention/Annotation_embedding_k_weights;multi_level__block_attention/Annotation_embedding_v_weights'multi_level__block_attention/query_bias%multi_level__block_attention/key_bias'multi_level__block_attention/value_bias8multi_level__block_attention/Epigenome_embedding_weights(multi_level__block_attention/output_biasconv1d/kerneldense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_317518
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
z
StaticRegexFullMatchStaticRegexFullMatchsaver_filename"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
\
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
a
Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
h
SelectSelectStaticRegexFullMatchConst_1Const_2"/device:CPU:**
T0*
_output_shapes
: 
`

StringJoin
StringJoinsaver_filenameSelect"/device:CPU:**
N*
_output_shapes
: 
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
┌
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*Г
value∙BЎ&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Annotation_embedding_q_weights/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Annotation_embedding_k_weights/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Annotation_embedding_v_weights/.ATTRIBUTES/VARIABLE_VALUEBKlayer_with_weights-2/Epigenome_embedding_weights/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/query_bias/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-2/key_bias/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/value_bias/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/output_bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBllayer_with_weights-2/Annotation_embedding_q_weights/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBllayer_with_weights-2/Annotation_embedding_k_weights/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBllayer_with_weights-2/Annotation_embedding_v_weights/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBilayer_with_weights-2/Epigenome_embedding_weights/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/query_bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/key_bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/value_bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/output_bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
╣
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ш
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices.locally_connected1d/kernel/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpOmulti_level__block_attention/Annotation_embedding_q_weights/Read/ReadVariableOpOmulti_level__block_attention/Annotation_embedding_k_weights/Read/ReadVariableOpOmulti_level__block_attention/Annotation_embedding_v_weights/Read/ReadVariableOpLmulti_level__block_attention/Epigenome_embedding_weights/Read/ReadVariableOp;multi_level__block_attention/query_bias/Read/ReadVariableOp9multi_level__block_attention/key_bias/Read/ReadVariableOp;multi_level__block_attention/value_bias/Read/ReadVariableOp<multi_level__block_attention/output_bias/Read/ReadVariableOp!conv1d/kernel/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOprho/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp:RMSprop/locally_connected1d/kernel/rms/Read/ReadVariableOp,RMSprop/dense/kernel/rms/Read/ReadVariableOp*RMSprop/dense/bias/rms/Read/ReadVariableOp[RMSprop/multi_level__block_attention/Annotation_embedding_q_weights/rms/Read/ReadVariableOp[RMSprop/multi_level__block_attention/Annotation_embedding_k_weights/rms/Read/ReadVariableOp[RMSprop/multi_level__block_attention/Annotation_embedding_v_weights/rms/Read/ReadVariableOpXRMSprop/multi_level__block_attention/Epigenome_embedding_weights/rms/Read/ReadVariableOpGRMSprop/multi_level__block_attention/query_bias/rms/Read/ReadVariableOpERMSprop/multi_level__block_attention/key_bias/rms/Read/ReadVariableOpGRMSprop/multi_level__block_attention/value_bias/rms/Read/ReadVariableOpHRMSprop/multi_level__block_attention/output_bias/rms/Read/ReadVariableOp-RMSprop/conv1d/kernel/rms/Read/ReadVariableOp.RMSprop/dense_1/kernel/rms/Read/ReadVariableOp,RMSprop/dense_1/bias/rms/Read/ReadVariableOpConst"/device:CPU:0*4
dtypes*
(2&	
Е
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
o
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixessaver_filename"/device:CPU:0
i
IdentityIdentitysaver_filename^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
▌
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*Г
value∙BЎ&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Annotation_embedding_q_weights/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Annotation_embedding_k_weights/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Annotation_embedding_v_weights/.ATTRIBUTES/VARIABLE_VALUEBKlayer_with_weights-2/Epigenome_embedding_weights/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/query_bias/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-2/key_bias/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/value_bias/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/output_bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBllayer_with_weights-2/Annotation_embedding_q_weights/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBllayer_with_weights-2/Annotation_embedding_k_weights/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBllayer_with_weights-2/Annotation_embedding_v_weights/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBilayer_with_weights-2/Epigenome_embedding_weights/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/query_bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/key_bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/value_bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/output_bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
╝
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
╨
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*о
_output_shapesЫ
Ш::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	
S

Identity_1Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOpAssignVariableOplocally_connected1d/kernel
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_1AssignVariableOpdense/kernel
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_2AssignVariableOp
dense/bias
Identity_3"/device:CPU:0*
dtype0
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
Л
AssignVariableOp_3AssignVariableOp;multi_level__block_attention/Annotation_embedding_q_weights
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
Л
AssignVariableOp_4AssignVariableOp;multi_level__block_attention/Annotation_embedding_k_weights
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
Л
AssignVariableOp_5AssignVariableOp;multi_level__block_attention/Annotation_embedding_v_weights
Identity_6"/device:CPU:0*
dtype0
U

Identity_7IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
И
AssignVariableOp_6AssignVariableOp8multi_level__block_attention/Epigenome_embedding_weights
Identity_7"/device:CPU:0*
dtype0
U

Identity_8IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
w
AssignVariableOp_7AssignVariableOp'multi_level__block_attention/query_bias
Identity_8"/device:CPU:0*
dtype0
U

Identity_9IdentityRestoreV2:8"/device:CPU:0*
T0*
_output_shapes
:
u
AssignVariableOp_8AssignVariableOp%multi_level__block_attention/key_bias
Identity_9"/device:CPU:0*
dtype0
V
Identity_10IdentityRestoreV2:9"/device:CPU:0*
T0*
_output_shapes
:
x
AssignVariableOp_9AssignVariableOp'multi_level__block_attention/value_biasIdentity_10"/device:CPU:0*
dtype0
W
Identity_11IdentityRestoreV2:10"/device:CPU:0*
T0*
_output_shapes
:
z
AssignVariableOp_10AssignVariableOp(multi_level__block_attention/output_biasIdentity_11"/device:CPU:0*
dtype0
W
Identity_12IdentityRestoreV2:11"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_11AssignVariableOpconv1d/kernelIdentity_12"/device:CPU:0*
dtype0
W
Identity_13IdentityRestoreV2:12"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_12AssignVariableOpdense_1/kernelIdentity_13"/device:CPU:0*
dtype0
W
Identity_14IdentityRestoreV2:13"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_13AssignVariableOpdense_1/biasIdentity_14"/device:CPU:0*
dtype0
W
Identity_15IdentityRestoreV2:14"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_14AssignVariableOpdecayIdentity_15"/device:CPU:0*
dtype0
W
Identity_16IdentityRestoreV2:15"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_15AssignVariableOplearning_rateIdentity_16"/device:CPU:0*
dtype0
W
Identity_17IdentityRestoreV2:16"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_16AssignVariableOpmomentumIdentity_17"/device:CPU:0*
dtype0
W
Identity_18IdentityRestoreV2:17"/device:CPU:0*
T0*
_output_shapes
:
U
AssignVariableOp_17AssignVariableOprhoIdentity_18"/device:CPU:0*
dtype0
W
Identity_19IdentityRestoreV2:18"/device:CPU:0*
T0	*
_output_shapes
:
^
AssignVariableOp_18AssignVariableOpRMSprop/iterIdentity_19"/device:CPU:0*
dtype0	
W
Identity_20IdentityRestoreV2:19"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_19AssignVariableOptotal_1Identity_20"/device:CPU:0*
dtype0
W
Identity_21IdentityRestoreV2:20"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_20AssignVariableOpcount_1Identity_21"/device:CPU:0*
dtype0
W
Identity_22IdentityRestoreV2:21"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_21AssignVariableOptotalIdentity_22"/device:CPU:0*
dtype0
W
Identity_23IdentityRestoreV2:22"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_22AssignVariableOpcountIdentity_23"/device:CPU:0*
dtype0
W
Identity_24IdentityRestoreV2:23"/device:CPU:0*
T0*
_output_shapes
:
x
AssignVariableOp_23AssignVariableOp&RMSprop/locally_connected1d/kernel/rmsIdentity_24"/device:CPU:0*
dtype0
W
Identity_25IdentityRestoreV2:24"/device:CPU:0*
T0*
_output_shapes
:
j
AssignVariableOp_24AssignVariableOpRMSprop/dense/kernel/rmsIdentity_25"/device:CPU:0*
dtype0
W
Identity_26IdentityRestoreV2:25"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_25AssignVariableOpRMSprop/dense/bias/rmsIdentity_26"/device:CPU:0*
dtype0
W
Identity_27IdentityRestoreV2:26"/device:CPU:0*
T0*
_output_shapes
:
Щ
AssignVariableOp_26AssignVariableOpGRMSprop/multi_level__block_attention/Annotation_embedding_q_weights/rmsIdentity_27"/device:CPU:0*
dtype0
W
Identity_28IdentityRestoreV2:27"/device:CPU:0*
T0*
_output_shapes
:
Щ
AssignVariableOp_27AssignVariableOpGRMSprop/multi_level__block_attention/Annotation_embedding_k_weights/rmsIdentity_28"/device:CPU:0*
dtype0
W
Identity_29IdentityRestoreV2:28"/device:CPU:0*
T0*
_output_shapes
:
Щ
AssignVariableOp_28AssignVariableOpGRMSprop/multi_level__block_attention/Annotation_embedding_v_weights/rmsIdentity_29"/device:CPU:0*
dtype0
W
Identity_30IdentityRestoreV2:29"/device:CPU:0*
T0*
_output_shapes
:
Ц
AssignVariableOp_29AssignVariableOpDRMSprop/multi_level__block_attention/Epigenome_embedding_weights/rmsIdentity_30"/device:CPU:0*
dtype0
W
Identity_31IdentityRestoreV2:30"/device:CPU:0*
T0*
_output_shapes
:
Е
AssignVariableOp_30AssignVariableOp3RMSprop/multi_level__block_attention/query_bias/rmsIdentity_31"/device:CPU:0*
dtype0
W
Identity_32IdentityRestoreV2:31"/device:CPU:0*
T0*
_output_shapes
:
Г
AssignVariableOp_31AssignVariableOp1RMSprop/multi_level__block_attention/key_bias/rmsIdentity_32"/device:CPU:0*
dtype0
W
Identity_33IdentityRestoreV2:32"/device:CPU:0*
T0*
_output_shapes
:
Е
AssignVariableOp_32AssignVariableOp3RMSprop/multi_level__block_attention/value_bias/rmsIdentity_33"/device:CPU:0*
dtype0
W
Identity_34IdentityRestoreV2:33"/device:CPU:0*
T0*
_output_shapes
:
Ж
AssignVariableOp_33AssignVariableOp4RMSprop/multi_level__block_attention/output_bias/rmsIdentity_34"/device:CPU:0*
dtype0
W
Identity_35IdentityRestoreV2:34"/device:CPU:0*
T0*
_output_shapes
:
k
AssignVariableOp_34AssignVariableOpRMSprop/conv1d/kernel/rmsIdentity_35"/device:CPU:0*
dtype0
W
Identity_36IdentityRestoreV2:35"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_35AssignVariableOpRMSprop/dense_1/kernel/rmsIdentity_36"/device:CPU:0*
dtype0
W
Identity_37IdentityRestoreV2:36"/device:CPU:0*
T0*
_output_shapes
:
j
AssignVariableOp_36AssignVariableOpRMSprop/dense_1/bias/rmsIdentity_37"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
В
Identity_38Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: Ш╒
ё
U
9__inference_global_average_pooling1d_layer_call_fn_318470

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
б
P
$__inference_add_layer_call_fn_318526
inputs_0
inputs_1
identityR
addAddV2inputs_0inputs_1*
T0*'
_output_shapes
:         O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         :Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
─╬
╦
A__inference_model_layer_call_and_return_conditional_losses_317322
input_layer_1H
2locally_connected1d_matmul_readvariableop_resource:

9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:M
;multi_level__block_attention_matmul_readvariableop_resource:O
=multi_level__block_attention_matmul_1_readvariableop_resource:O
=multi_level__block_attention_matmul_2_readvariableop_resource:J
8multi_level__block_attention_add_readvariableop_resource:
L
:multi_level__block_attention_add_1_readvariableop_resource:
L
:multi_level__block_attention_add_2_readvariableop_resource:
J
8multi_level__block_attention_mul_readvariableop_resource:

L
:multi_level__block_attention_add_3_readvariableop_resource:
H
2conv1d_conv1d_expanddims_1_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:
5
'dense_1_biasadd_readvariableop_resource:
identityИв)conv1d/Conv1D/ExpandDims_1/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/Tensordot/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв)locally_connected1d/MatMul/ReadVariableOpв2multi_level__block_attention/MatMul/ReadVariableOpв4multi_level__block_attention/MatMul_1/ReadVariableOpв4multi_level__block_attention/MatMul_2/ReadVariableOpв/multi_level__block_attention/Mul/ReadVariableOpв/multi_level__block_attention/add/ReadVariableOpв1multi_level__block_attention/add_1/ReadVariableOpв1multi_level__block_attention/add_2/ReadVariableOpв1multi_level__block_attention/add_3/ReadVariableOpД
zero_padding1d/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Д
zero_padding1d/PadPadinput_layer_1$zero_padding1d/Pad/paddings:output:0*
T0*+
_output_shapes
:         e|
'locally_connected1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            ~
)locally_connected1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       ~
)locally_connected1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ═
!locally_connected1d/strided_sliceStridedSlicezero_padding1d/Pad:output:00locally_connected1d/strided_slice/stack:output:02locally_connected1d/strided_slice/stack_1:output:02locally_connected1d/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskv
!locally_connected1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ┤
locally_connected1d/ReshapeReshape*locally_connected1d/strided_slice:output:0*locally_connected1d/Reshape/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       А
+locally_connected1d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_1StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_1/stack:output:04locally_connected1d/strided_slice_1/stack_1:output:04locally_connected1d/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_1Reshape,locally_connected1d/strided_slice_1:output:0,locally_connected1d/Reshape_1/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_2StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_2/stack:output:04locally_connected1d/strided_slice_2/stack_1:output:04locally_connected1d/strided_slice_2/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_2Reshape,locally_connected1d/strided_slice_2:output:0,locally_connected1d/Reshape_2/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    (       А
+locally_connected1d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_3StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_3/stack:output:04locally_connected1d/strided_slice_3/stack_1:output:04locally_connected1d/strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_3Reshape,locally_connected1d/strided_slice_3:output:0,locally_connected1d/Reshape_3/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"    (       А
+locally_connected1d/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    2       А
+locally_connected1d/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_4StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_4/stack:output:04locally_connected1d/strided_slice_4/stack_1:output:04locally_connected1d/strided_slice_4/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_4Reshape,locally_connected1d/strided_slice_4:output:0,locally_connected1d/Reshape_4/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*!
valueB"    2       А
+locally_connected1d/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    <       А
+locally_connected1d/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_5StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_5/stack:output:04locally_connected1d/strided_slice_5/stack_1:output:04locally_connected1d/strided_slice_5/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_5Reshape,locally_connected1d/strided_slice_5:output:0,locally_connected1d/Reshape_5/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"    <       А
+locally_connected1d/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    F       А
+locally_connected1d/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_6StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_6/stack:output:04locally_connected1d/strided_slice_6/stack_1:output:04locally_connected1d/strided_slice_6/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_6Reshape,locally_connected1d/strided_slice_6:output:0,locally_connected1d/Reshape_6/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*!
valueB"    F       А
+locally_connected1d/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    P       А
+locally_connected1d/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_7StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_7/stack:output:04locally_connected1d/strided_slice_7/stack_1:output:04locally_connected1d/strided_slice_7/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_7Reshape,locally_connected1d/strided_slice_7:output:0,locally_connected1d/Reshape_7/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*!
valueB"    P       А
+locally_connected1d/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    Z       А
+locally_connected1d/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_8StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_8/stack:output:04locally_connected1d/strided_slice_8/stack_1:output:04locally_connected1d/strided_slice_8/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_8Reshape,locally_connected1d/strided_slice_8:output:0,locally_connected1d/Reshape_8/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*!
valueB"    Z       А
+locally_connected1d/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    d       А
+locally_connected1d/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_9StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_9/stack:output:04locally_connected1d/strided_slice_9/stack_1:output:04locally_connected1d/strided_slice_9/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_9Reshape,locally_connected1d/strided_slice_9:output:0,locally_connected1d/Reshape_9/shape:output:0*
T0*+
_output_shapes
:         
a
locally_connected1d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Э
locally_connected1d/concatConcatV2$locally_connected1d/Reshape:output:0&locally_connected1d/Reshape_1:output:0&locally_connected1d/Reshape_2:output:0&locally_connected1d/Reshape_3:output:0&locally_connected1d/Reshape_4:output:0&locally_connected1d/Reshape_5:output:0&locally_connected1d/Reshape_6:output:0&locally_connected1d/Reshape_7:output:0&locally_connected1d/Reshape_8:output:0&locally_connected1d/Reshape_9:output:0(locally_connected1d/concat/axis:output:0*
N
*
T0*+
_output_shapes
:
         
а
)locally_connected1d/MatMul/ReadVariableOpReadVariableOp2locally_connected1d_matmul_readvariableop_resource*"
_output_shapes
:

*
dtype0╣
locally_connected1d/MatMulBatchMatMulV2#locally_connected1d/concat:output:01locally_connected1d/MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:
         l
locally_connected1d/ShapeShape#locally_connected1d/MatMul:output:0*
T0*
_output_shapes
:y
$locally_connected1d/Reshape_10/shapeConst*
_output_shapes
:*
dtype0*!
valueB"
          │
locally_connected1d/Reshape_10Reshape#locally_connected1d/MatMul:output:0-locally_connected1d/Reshape_10/shape:output:0*
T0*+
_output_shapes
:
         w
"locally_connected1d/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╢
locally_connected1d/transpose	Transpose'locally_connected1d/Reshape_10:output:0+locally_connected1d/transpose/perm:output:0*
T0*+
_output_shapes
:         
Ж
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       f
dense/Tensordot/ShapeShape!locally_connected1d/transpose:y:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╙
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ж
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ┤
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Л
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:а
dense/Tensordot/transpose	Transpose!locally_connected1d/transpose:y:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:         
Ь
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  Ь
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Х
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
о
2multi_level__block_attention/MatMul/ReadVariableOpReadVariableOp;multi_level__block_attention_matmul_readvariableop_resource*
_output_shapes

:*
dtype0╛
#multi_level__block_attention/MatMulBatchMatMulV2dense/BiasAdd:output:0:multi_level__block_attention/MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
▓
4multi_level__block_attention/MatMul_1/ReadVariableOpReadVariableOp=multi_level__block_attention_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0┬
%multi_level__block_attention/MatMul_1BatchMatMulV2dense/BiasAdd:output:0<multi_level__block_attention/MatMul_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
▓
4multi_level__block_attention/MatMul_2/ReadVariableOpReadVariableOp=multi_level__block_attention_matmul_2_readvariableop_resource*
_output_shapes

:*
dtype0┬
%multi_level__block_attention/MatMul_2BatchMatMulV2dense/BiasAdd:output:0<multi_level__block_attention/MatMul_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
и
/multi_level__block_attention/add/ReadVariableOpReadVariableOp8multi_level__block_attention_add_readvariableop_resource*
_output_shapes

:
*
dtype0╞
 multi_level__block_attention/addAddV2,multi_level__block_attention/MatMul:output:07multi_level__block_attention/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_1/ReadVariableOpReadVariableOp:multi_level__block_attention_add_1_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_1AddV2.multi_level__block_attention/MatMul_1:output:09multi_level__block_attention/add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_2/ReadVariableOpReadVariableOp:multi_level__block_attention_add_2_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_2AddV2.multi_level__block_attention/MatMul_2:output:09multi_level__block_attention/add_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
╟
%multi_level__block_attention/MatMul_3BatchMatMulV2$multi_level__block_attention/add:z:0&multi_level__block_attention/add_1:z:0*
T0*+
_output_shapes
:         

*
adj_y(e
#multi_level__block_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
З
!multi_level__block_attention/CastCast,multi_level__block_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: q
!multi_level__block_attention/SqrtSqrt%multi_level__block_attention/Cast:y:0*
T0*
_output_shapes
: ╝
$multi_level__block_attention/truedivRealDiv.multi_level__block_attention/MatMul_3:output:0%multi_level__block_attention/Sqrt:y:0*
T0*+
_output_shapes
:         

П
$multi_level__block_attention/SoftmaxSoftmax(multi_level__block_attention/truediv:z:0*
T0*+
_output_shapes
:         

и
/multi_level__block_attention/Mul/ReadVariableOpReadVariableOp8multi_level__block_attention_mul_readvariableop_resource*
_output_shapes

:

*
dtype0╞
 multi_level__block_attention/MulMul.multi_level__block_attention/Softmax:softmax:07multi_level__block_attention/Mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         

║
%multi_level__block_attention/MatMul_4BatchMatMulV2$multi_level__block_attention/Mul:z:0&multi_level__block_attention/add_2:z:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_3/ReadVariableOpReadVariableOp:multi_level__block_attention_add_3_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_3AddV2.multi_level__block_attention/MatMul_4:output:09multi_level__block_attention/add_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        п
conv1d/Conv1D/ExpandDims
ExpandDims&multi_level__block_attention/add_3:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
а
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╡
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:┴
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
О
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        s
activation/SigmoidSigmoidconv1d/Conv1D/Squeeze:output:0*
T0*+
_output_shapes
:         
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   |
flatten/ReshapeReshapeactivation/Sigmoid:y:0flatten/Const:output:0*
T0*'
_output_shapes
:         
q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :▒
global_average_pooling1d/MeanMeanconv1d/Conv1D/Squeeze:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         `
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Р
flatten_1/ReshapeReshape&global_average_pooling1d/Mean:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:         Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Л
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x
add/addAddV2flatten_1/Reshape:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentityadd/add:z:0^NoOp*
T0*'
_output_shapes
:         ┬
NoOpNoOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*^locally_connected1d/MatMul/ReadVariableOp3^multi_level__block_attention/MatMul/ReadVariableOp5^multi_level__block_attention/MatMul_1/ReadVariableOp5^multi_level__block_attention/MatMul_2/ReadVariableOp0^multi_level__block_attention/Mul/ReadVariableOp0^multi_level__block_attention/add/ReadVariableOp2^multi_level__block_attention/add_1/ReadVariableOp2^multi_level__block_attention/add_2/ReadVariableOp2^multi_level__block_attention/add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         d: : : : : : : : : : : : : : 2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2V
)locally_connected1d/MatMul/ReadVariableOp)locally_connected1d/MatMul/ReadVariableOp2h
2multi_level__block_attention/MatMul/ReadVariableOp2multi_level__block_attention/MatMul/ReadVariableOp2l
4multi_level__block_attention/MatMul_1/ReadVariableOp4multi_level__block_attention/MatMul_1/ReadVariableOp2l
4multi_level__block_attention/MatMul_2/ReadVariableOp4multi_level__block_attention/MatMul_2/ReadVariableOp2b
/multi_level__block_attention/Mul/ReadVariableOp/multi_level__block_attention/Mul/ReadVariableOp2b
/multi_level__block_attention/add/ReadVariableOp/multi_level__block_attention/add/ReadVariableOp2f
1multi_level__block_attention/add_1/ReadVariableOp1multi_level__block_attention/add_1/ReadVariableOp2f
1multi_level__block_attention/add_2/ReadVariableOp1multi_level__block_attention/add_2/ReadVariableOp2f
1multi_level__block_attention/add_3/ReadVariableOp1multi_level__block_attention/add_3/ReadVariableOp:Z V
+
_output_shapes
:         d
'
_user_specified_nameinput_layer_1
╗
_
C__inference_flatten_layer_call_and_return_conditional_losses_318488

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╡
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_318500

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ы!
п
=__inference_multi_level__block_attention_layer_call_fn_318394
x_00
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:2
 matmul_2_readvariableop_resource:-
add_readvariableop_resource:
/
add_1_readvariableop_resource:
/
add_2_readvariableop_resource:
-
mul_readvariableop_resource:

/
add_3_readvariableop_resource:

identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвMatMul_2/ReadVariableOpвMul/ReadVariableOpвadd/ReadVariableOpвadd_1/ReadVariableOpвadd_2/ReadVariableOpвadd_3/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0q
MatMulBatchMatMulV2x_0MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0u
MatMul_1BatchMatMulV2x_0MatMul_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
x
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype0u
MatMul_2BatchMatMulV2x_0MatMul_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
n
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

:
*
dtype0o
addAddV2MatMul:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
r
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes

:
*
dtype0u
add_1AddV2MatMul_1:output:0add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
r
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes

:
*
dtype0u
add_2AddV2MatMul_2:output:0add_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
p
MatMul_3BatchMatMulV2add:z:0	add_1:z:0*
T0*+
_output_shapes
:         

*
adj_y(H
Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 7
SqrtSqrtCast:y:0*
T0*
_output_shapes
: e
truedivRealDivMatMul_3:output:0Sqrt:y:0*
T0*+
_output_shapes
:         

U
SoftmaxSoftmaxtruediv:z:0*
T0*+
_output_shapes
:         

n
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

:

*
dtype0o
MulMulSoftmax:softmax:0Mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         

c
MatMul_4BatchMatMulV2Mul:z:0	add_2:z:0*
T0*+
_output_shapes
:         
r
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes

:
*
dtype0u
add_3AddV2MatMul_4:output:0add_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
\
IdentityIdentity	add_3:z:0^NoOp*
T0*+
_output_shapes
:         
`

Identity_1Identitytruediv:z:0^NoOp*
T0*+
_output_shapes
:         

Б
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^Mul/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp^add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         
: : : : : : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2,
add_3/ReadVariableOpadd_3/ReadVariableOp:P L
+
_output_shapes
:         


_user_specified_namex/0
¤▀
╙
!__inference__wrapped_model_315819
input_layer_1N
8model_locally_connected1d_matmul_readvariableop_resource:

?
-model_dense_tensordot_readvariableop_resource:9
+model_dense_biasadd_readvariableop_resource:S
Amodel_multi_level__block_attention_matmul_readvariableop_resource:U
Cmodel_multi_level__block_attention_matmul_1_readvariableop_resource:U
Cmodel_multi_level__block_attention_matmul_2_readvariableop_resource:P
>model_multi_level__block_attention_add_readvariableop_resource:
R
@model_multi_level__block_attention_add_1_readvariableop_resource:
R
@model_multi_level__block_attention_add_2_readvariableop_resource:
P
>model_multi_level__block_attention_mul_readvariableop_resource:

R
@model_multi_level__block_attention_add_3_readvariableop_resource:
N
8model_conv1d_conv1d_expanddims_1_readvariableop_resource:>
,model_dense_1_matmul_readvariableop_resource:
;
-model_dense_1_biasadd_readvariableop_resource:
identityИв/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOpв"model/dense/BiasAdd/ReadVariableOpв$model/dense/Tensordot/ReadVariableOpв$model/dense_1/BiasAdd/ReadVariableOpв#model/dense_1/MatMul/ReadVariableOpв/model/locally_connected1d/MatMul/ReadVariableOpв8model/multi_level__block_attention/MatMul/ReadVariableOpв:model/multi_level__block_attention/MatMul_1/ReadVariableOpв:model/multi_level__block_attention/MatMul_2/ReadVariableOpв5model/multi_level__block_attention/Mul/ReadVariableOpв5model/multi_level__block_attention/add/ReadVariableOpв7model/multi_level__block_attention/add_1/ReadVariableOpв7model/multi_level__block_attention/add_2/ReadVariableOpв7model/multi_level__block_attention/add_3/ReadVariableOpК
!model/zero_padding1d/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Р
model/zero_padding1d/PadPadinput_layer_1*model/zero_padding1d/Pad/paddings:output:0*
T0*+
_output_shapes
:         eВ
-model/locally_connected1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            Д
/model/locally_connected1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       Д
/model/locally_connected1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ы
'model/locally_connected1d/strided_sliceStridedSlice!model/zero_padding1d/Pad:output:06model/locally_connected1d/strided_slice/stack:output:08model/locally_connected1d/strided_slice/stack_1:output:08model/locally_connected1d/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_mask|
'model/locally_connected1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ╞
!model/locally_connected1d/ReshapeReshape0model/locally_connected1d/strided_slice:output:00model/locally_connected1d/Reshape/shape:output:0*
T0*+
_output_shapes
:         
Д
/model/locally_connected1d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       Ж
1model/locally_connected1d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           Ж
1model/locally_connected1d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         є
)model/locally_connected1d/strided_slice_1StridedSlice!model/zero_padding1d/Pad:output:08model/locally_connected1d/strided_slice_1/stack:output:0:model/locally_connected1d/strided_slice_1/stack_1:output:0:model/locally_connected1d/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_mask~
)model/locally_connected1d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ╠
#model/locally_connected1d/Reshape_1Reshape2model/locally_connected1d/strided_slice_1:output:02model/locally_connected1d/Reshape_1/shape:output:0*
T0*+
_output_shapes
:         
Д
/model/locally_connected1d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           Ж
1model/locally_connected1d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           Ж
1model/locally_connected1d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         є
)model/locally_connected1d/strided_slice_2StridedSlice!model/zero_padding1d/Pad:output:08model/locally_connected1d/strided_slice_2/stack:output:0:model/locally_connected1d/strided_slice_2/stack_1:output:0:model/locally_connected1d/strided_slice_2/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_mask~
)model/locally_connected1d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ╠
#model/locally_connected1d/Reshape_2Reshape2model/locally_connected1d/strided_slice_2:output:02model/locally_connected1d/Reshape_2/shape:output:0*
T0*+
_output_shapes
:         
Д
/model/locally_connected1d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"           Ж
1model/locally_connected1d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    (       Ж
1model/locally_connected1d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         є
)model/locally_connected1d/strided_slice_3StridedSlice!model/zero_padding1d/Pad:output:08model/locally_connected1d/strided_slice_3/stack:output:0:model/locally_connected1d/strided_slice_3/stack_1:output:0:model/locally_connected1d/strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_mask~
)model/locally_connected1d/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ╠
#model/locally_connected1d/Reshape_3Reshape2model/locally_connected1d/strided_slice_3:output:02model/locally_connected1d/Reshape_3/shape:output:0*
T0*+
_output_shapes
:         
Д
/model/locally_connected1d/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"    (       Ж
1model/locally_connected1d/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    2       Ж
1model/locally_connected1d/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         є
)model/locally_connected1d/strided_slice_4StridedSlice!model/zero_padding1d/Pad:output:08model/locally_connected1d/strided_slice_4/stack:output:0:model/locally_connected1d/strided_slice_4/stack_1:output:0:model/locally_connected1d/strided_slice_4/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_mask~
)model/locally_connected1d/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ╠
#model/locally_connected1d/Reshape_4Reshape2model/locally_connected1d/strided_slice_4:output:02model/locally_connected1d/Reshape_4/shape:output:0*
T0*+
_output_shapes
:         
Д
/model/locally_connected1d/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*!
valueB"    2       Ж
1model/locally_connected1d/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    <       Ж
1model/locally_connected1d/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         є
)model/locally_connected1d/strided_slice_5StridedSlice!model/zero_padding1d/Pad:output:08model/locally_connected1d/strided_slice_5/stack:output:0:model/locally_connected1d/strided_slice_5/stack_1:output:0:model/locally_connected1d/strided_slice_5/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_mask~
)model/locally_connected1d/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ╠
#model/locally_connected1d/Reshape_5Reshape2model/locally_connected1d/strided_slice_5:output:02model/locally_connected1d/Reshape_5/shape:output:0*
T0*+
_output_shapes
:         
Д
/model/locally_connected1d/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"    <       Ж
1model/locally_connected1d/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    F       Ж
1model/locally_connected1d/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         є
)model/locally_connected1d/strided_slice_6StridedSlice!model/zero_padding1d/Pad:output:08model/locally_connected1d/strided_slice_6/stack:output:0:model/locally_connected1d/strided_slice_6/stack_1:output:0:model/locally_connected1d/strided_slice_6/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_mask~
)model/locally_connected1d/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ╠
#model/locally_connected1d/Reshape_6Reshape2model/locally_connected1d/strided_slice_6:output:02model/locally_connected1d/Reshape_6/shape:output:0*
T0*+
_output_shapes
:         
Д
/model/locally_connected1d/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*!
valueB"    F       Ж
1model/locally_connected1d/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    P       Ж
1model/locally_connected1d/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         є
)model/locally_connected1d/strided_slice_7StridedSlice!model/zero_padding1d/Pad:output:08model/locally_connected1d/strided_slice_7/stack:output:0:model/locally_connected1d/strided_slice_7/stack_1:output:0:model/locally_connected1d/strided_slice_7/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_mask~
)model/locally_connected1d/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ╠
#model/locally_connected1d/Reshape_7Reshape2model/locally_connected1d/strided_slice_7:output:02model/locally_connected1d/Reshape_7/shape:output:0*
T0*+
_output_shapes
:         
Д
/model/locally_connected1d/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*!
valueB"    P       Ж
1model/locally_connected1d/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    Z       Ж
1model/locally_connected1d/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         є
)model/locally_connected1d/strided_slice_8StridedSlice!model/zero_padding1d/Pad:output:08model/locally_connected1d/strided_slice_8/stack:output:0:model/locally_connected1d/strided_slice_8/stack_1:output:0:model/locally_connected1d/strided_slice_8/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_mask~
)model/locally_connected1d/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ╠
#model/locally_connected1d/Reshape_8Reshape2model/locally_connected1d/strided_slice_8:output:02model/locally_connected1d/Reshape_8/shape:output:0*
T0*+
_output_shapes
:         
Д
/model/locally_connected1d/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*!
valueB"    Z       Ж
1model/locally_connected1d/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    d       Ж
1model/locally_connected1d/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         є
)model/locally_connected1d/strided_slice_9StridedSlice!model/zero_padding1d/Pad:output:08model/locally_connected1d/strided_slice_9/stack:output:0:model/locally_connected1d/strided_slice_9/stack_1:output:0:model/locally_connected1d/strided_slice_9/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_mask~
)model/locally_connected1d/Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ╠
#model/locally_connected1d/Reshape_9Reshape2model/locally_connected1d/strided_slice_9:output:02model/locally_connected1d/Reshape_9/shape:output:0*
T0*+
_output_shapes
:         
g
%model/locally_connected1d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : х
 model/locally_connected1d/concatConcatV2*model/locally_connected1d/Reshape:output:0,model/locally_connected1d/Reshape_1:output:0,model/locally_connected1d/Reshape_2:output:0,model/locally_connected1d/Reshape_3:output:0,model/locally_connected1d/Reshape_4:output:0,model/locally_connected1d/Reshape_5:output:0,model/locally_connected1d/Reshape_6:output:0,model/locally_connected1d/Reshape_7:output:0,model/locally_connected1d/Reshape_8:output:0,model/locally_connected1d/Reshape_9:output:0.model/locally_connected1d/concat/axis:output:0*
N
*
T0*+
_output_shapes
:
         
м
/model/locally_connected1d/MatMul/ReadVariableOpReadVariableOp8model_locally_connected1d_matmul_readvariableop_resource*"
_output_shapes
:

*
dtype0╦
 model/locally_connected1d/MatMulBatchMatMulV2)model/locally_connected1d/concat:output:07model/locally_connected1d/MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:
         x
model/locally_connected1d/ShapeShape)model/locally_connected1d/MatMul:output:0*
T0*
_output_shapes
:
*model/locally_connected1d/Reshape_10/shapeConst*
_output_shapes
:*
dtype0*!
valueB"
          ┼
$model/locally_connected1d/Reshape_10Reshape)model/locally_connected1d/MatMul:output:03model/locally_connected1d/Reshape_10/shape:output:0*
T0*+
_output_shapes
:
         }
(model/locally_connected1d/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╚
#model/locally_connected1d/transpose	Transpose-model/locally_connected1d/Reshape_10:output:01model/locally_connected1d/transpose/perm:output:0*
T0*+
_output_shapes
:         
Т
$model/dense/Tensordot/ReadVariableOpReadVariableOp-model_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0d
model/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
model/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       r
model/dense/Tensordot/ShapeShape'model/locally_connected1d/transpose:y:0*
T0*
_output_shapes
:e
#model/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ы
model/dense/Tensordot/GatherV2GatherV2$model/dense/Tensordot/Shape:output:0#model/dense/Tensordot/free:output:0,model/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
%model/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : я
 model/dense/Tensordot/GatherV2_1GatherV2$model/dense/Tensordot/Shape:output:0#model/dense/Tensordot/axes:output:0.model/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
model/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Т
model/dense/Tensordot/ProdProd'model/dense/Tensordot/GatherV2:output:0$model/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: g
model/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ш
model/dense/Tensordot/Prod_1Prod)model/dense/Tensordot/GatherV2_1:output:0&model/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: c
!model/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╠
model/dense/Tensordot/concatConcatV2#model/dense/Tensordot/free:output:0#model/dense/Tensordot/axes:output:0*model/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Э
model/dense/Tensordot/stackPack#model/dense/Tensordot/Prod:output:0%model/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:▓
model/dense/Tensordot/transpose	Transpose'model/locally_connected1d/transpose:y:0%model/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:         
о
model/dense/Tensordot/ReshapeReshape#model/dense/Tensordot/transpose:y:0$model/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  о
model/dense/Tensordot/MatMulMatMul&model/dense/Tensordot/Reshape:output:0,model/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         g
model/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:e
#model/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
model/dense/Tensordot/concat_1ConcatV2'model/dense/Tensordot/GatherV2:output:0&model/dense/Tensordot/Const_2:output:0,model/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:з
model/dense/TensordotReshape&model/dense/Tensordot/MatMul:product:0'model/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         
К
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0а
model/dense/BiasAddBiasAddmodel/dense/Tensordot:output:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
║
8model/multi_level__block_attention/MatMul/ReadVariableOpReadVariableOpAmodel_multi_level__block_attention_matmul_readvariableop_resource*
_output_shapes

:*
dtype0╨
)model/multi_level__block_attention/MatMulBatchMatMulV2model/dense/BiasAdd:output:0@model/multi_level__block_attention/MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
╛
:model/multi_level__block_attention/MatMul_1/ReadVariableOpReadVariableOpCmodel_multi_level__block_attention_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0╘
+model/multi_level__block_attention/MatMul_1BatchMatMulV2model/dense/BiasAdd:output:0Bmodel/multi_level__block_attention/MatMul_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
╛
:model/multi_level__block_attention/MatMul_2/ReadVariableOpReadVariableOpCmodel_multi_level__block_attention_matmul_2_readvariableop_resource*
_output_shapes

:*
dtype0╘
+model/multi_level__block_attention/MatMul_2BatchMatMulV2model/dense/BiasAdd:output:0Bmodel/multi_level__block_attention/MatMul_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
┤
5model/multi_level__block_attention/add/ReadVariableOpReadVariableOp>model_multi_level__block_attention_add_readvariableop_resource*
_output_shapes

:
*
dtype0╪
&model/multi_level__block_attention/addAddV22model/multi_level__block_attention/MatMul:output:0=model/multi_level__block_attention/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
╕
7model/multi_level__block_attention/add_1/ReadVariableOpReadVariableOp@model_multi_level__block_attention_add_1_readvariableop_resource*
_output_shapes

:
*
dtype0▐
(model/multi_level__block_attention/add_1AddV24model/multi_level__block_attention/MatMul_1:output:0?model/multi_level__block_attention/add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
╕
7model/multi_level__block_attention/add_2/ReadVariableOpReadVariableOp@model_multi_level__block_attention_add_2_readvariableop_resource*
_output_shapes

:
*
dtype0▐
(model/multi_level__block_attention/add_2AddV24model/multi_level__block_attention/MatMul_2:output:0?model/multi_level__block_attention/add_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
┘
+model/multi_level__block_attention/MatMul_3BatchMatMulV2*model/multi_level__block_attention/add:z:0,model/multi_level__block_attention/add_1:z:0*
T0*+
_output_shapes
:         

*
adj_y(k
)model/multi_level__block_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
У
'model/multi_level__block_attention/CastCast2model/multi_level__block_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: }
'model/multi_level__block_attention/SqrtSqrt+model/multi_level__block_attention/Cast:y:0*
T0*
_output_shapes
: ╬
*model/multi_level__block_attention/truedivRealDiv4model/multi_level__block_attention/MatMul_3:output:0+model/multi_level__block_attention/Sqrt:y:0*
T0*+
_output_shapes
:         

Ы
*model/multi_level__block_attention/SoftmaxSoftmax.model/multi_level__block_attention/truediv:z:0*
T0*+
_output_shapes
:         

┤
5model/multi_level__block_attention/Mul/ReadVariableOpReadVariableOp>model_multi_level__block_attention_mul_readvariableop_resource*
_output_shapes

:

*
dtype0╪
&model/multi_level__block_attention/MulMul4model/multi_level__block_attention/Softmax:softmax:0=model/multi_level__block_attention/Mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         

╠
+model/multi_level__block_attention/MatMul_4BatchMatMulV2*model/multi_level__block_attention/Mul:z:0,model/multi_level__block_attention/add_2:z:0*
T0*+
_output_shapes
:         
╕
7model/multi_level__block_attention/add_3/ReadVariableOpReadVariableOp@model_multi_level__block_attention_add_3_readvariableop_resource*
_output_shapes

:
*
dtype0▐
(model/multi_level__block_attention/add_3AddV24model/multi_level__block_attention/MatMul_4:output:0?model/multi_level__block_attention/add_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
m
"model/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        ┴
model/conv1d/Conv1D/ExpandDims
ExpandDims,model/multi_level__block_attention/add_3:z:0+model/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
м
/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp8model_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0f
$model/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╟
 model/conv1d/Conv1D/ExpandDims_1
ExpandDims7model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0-model/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:╙
model/conv1d/Conv1DConv2D'model/conv1d/Conv1D/ExpandDims:output:0)model/conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
Ъ
model/conv1d/Conv1D/SqueezeSqueezemodel/conv1d/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        
model/activation/SigmoidSigmoid$model/conv1d/Conv1D/Squeeze:output:0*
T0*+
_output_shapes
:         
d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   О
model/flatten/ReshapeReshapemodel/activation/Sigmoid:y:0model/flatten/Const:output:0*
T0*'
_output_shapes
:         
w
5model/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :├
#model/global_average_pooling1d/MeanMean$model/conv1d/Conv1D/Squeeze:output:0>model/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         f
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       в
model/flatten_1/ReshapeReshape,model/global_average_pooling1d/Mean:output:0model/flatten_1/Const:output:0*
T0*'
_output_shapes
:         Р
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Э
model/dense_1/MatMulMatMulmodel/flatten/Reshape:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         О
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0а
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         К
model/add/addAddV2 model/flatten_1/Reshape:output:0model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitymodel/add/add:z:0^NoOp*
T0*'
_output_shapes
:         Ц
NoOpNoOp0^model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp%^model/dense/Tensordot/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp0^model/locally_connected1d/MatMul/ReadVariableOp9^model/multi_level__block_attention/MatMul/ReadVariableOp;^model/multi_level__block_attention/MatMul_1/ReadVariableOp;^model/multi_level__block_attention/MatMul_2/ReadVariableOp6^model/multi_level__block_attention/Mul/ReadVariableOp6^model/multi_level__block_attention/add/ReadVariableOp8^model/multi_level__block_attention/add_1/ReadVariableOp8^model/multi_level__block_attention/add_2/ReadVariableOp8^model/multi_level__block_attention/add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         d: : : : : : : : : : : : : : 2b
/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp/model/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2L
$model/dense/Tensordot/ReadVariableOp$model/dense/Tensordot/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2b
/model/locally_connected1d/MatMul/ReadVariableOp/model/locally_connected1d/MatMul/ReadVariableOp2t
8model/multi_level__block_attention/MatMul/ReadVariableOp8model/multi_level__block_attention/MatMul/ReadVariableOp2x
:model/multi_level__block_attention/MatMul_1/ReadVariableOp:model/multi_level__block_attention/MatMul_1/ReadVariableOp2x
:model/multi_level__block_attention/MatMul_2/ReadVariableOp:model/multi_level__block_attention/MatMul_2/ReadVariableOp2n
5model/multi_level__block_attention/Mul/ReadVariableOp5model/multi_level__block_attention/Mul/ReadVariableOp2n
5model/multi_level__block_attention/add/ReadVariableOp5model/multi_level__block_attention/add/ReadVariableOp2r
7model/multi_level__block_attention/add_1/ReadVariableOp7model/multi_level__block_attention/add_1/ReadVariableOp2r
7model/multi_level__block_attention/add_2/ReadVariableOp7model/multi_level__block_attention/add_2/ReadVariableOp2r
7model/multi_level__block_attention/add_3/ReadVariableOp7model/multi_level__block_attention/add_3/ReadVariableOp:Z V
+
_output_shapes
:         d
'
_user_specified_nameinput_layer_1
░L
╝
O__inference_locally_connected1d_layer_call_and_return_conditional_losses_318298

inputs4
matmul_readvariableop_resource:


identityИвMatMul/ReadVariableOph
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ш
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskb
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   x
ReshapeReshapestrided_slice:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:         
j
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ё
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskd
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ~
	Reshape_1Reshapestrided_slice_1:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:         
j
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ё
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskd
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ~
	Reshape_2Reshapestrided_slice_2:output:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:         
j
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    (       l
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ё
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskd
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ~
	Reshape_3Reshapestrided_slice_3:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:         
j
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"    (       l
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    2       l
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ё
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskd
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ~
	Reshape_4Reshapestrided_slice_4:output:0Reshape_4/shape:output:0*
T0*+
_output_shapes
:         
j
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*!
valueB"    2       l
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    <       l
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ё
strided_slice_5StridedSliceinputsstrided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskd
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ~
	Reshape_5Reshapestrided_slice_5:output:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:         
j
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"    <       l
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    F       l
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ё
strided_slice_6StridedSliceinputsstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskd
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ~
	Reshape_6Reshapestrided_slice_6:output:0Reshape_6/shape:output:0*
T0*+
_output_shapes
:         
j
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*!
valueB"    F       l
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    P       l
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ё
strided_slice_7StridedSliceinputsstrided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskd
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ~
	Reshape_7Reshapestrided_slice_7:output:0Reshape_7/shape:output:0*
T0*+
_output_shapes
:         
j
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*!
valueB"    P       l
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    Z       l
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ё
strided_slice_8StridedSliceinputsstrided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskd
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ~
	Reshape_8Reshapestrided_slice_8:output:0Reshape_8/shape:output:0*
T0*+
_output_shapes
:         
j
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*!
valueB"    Z       l
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    d       l
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ё
strided_slice_9StridedSliceinputsstrided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskd
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ~
	Reshape_9Reshapestrided_slice_9:output:0Reshape_9/shape:output:0*
T0*+
_output_shapes
:         
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : н
concatConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0concat/axis:output:0*
N
*
T0*+
_output_shapes
:
         
x
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*"
_output_shapes
:

*
dtype0}
MatMulBatchMatMulV2concat:output:0MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:
         D
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:e
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*!
valueB"
          w

Reshape_10ReshapeMatMul:output:0Reshape_10/shape:output:0*
T0*+
_output_shapes
:
         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          z
	transpose	TransposeReshape_10:output:0transpose/perm:output:0*
T0*+
_output_shapes
:         
`
IdentityIdentitytranspose:y:0^NoOp*
T0*+
_output_shapes
:         
^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         e: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:S O
+
_output_shapes
:         e
 
_user_specified_nameinputs
м
f
J__inference_zero_padding1d_layer_call_and_return_conditional_losses_318150

inputs
identityu
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       q
PadPadinputsPad/paddings:output:0*
T0*=
_output_shapes+
):'                           j
IdentityIdentityPad:output:0*
T0*=
_output_shapes+
):'                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
╢!
╩
X__inference_multi_level__block_attention_layer_call_and_return_conditional_losses_318430
x_00
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:2
 matmul_2_readvariableop_resource:-
add_readvariableop_resource:
/
add_1_readvariableop_resource:
/
add_2_readvariableop_resource:
-
mul_readvariableop_resource:

/
add_3_readvariableop_resource:

identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвMatMul_2/ReadVariableOpвMul/ReadVariableOpвadd/ReadVariableOpвadd_1/ReadVariableOpвadd_2/ReadVariableOpвadd_3/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0q
MatMulBatchMatMulV2x_0MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0u
MatMul_1BatchMatMulV2x_0MatMul_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
x
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:*
dtype0u
MatMul_2BatchMatMulV2x_0MatMul_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
n
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes

:
*
dtype0o
addAddV2MatMul:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
r
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes

:
*
dtype0u
add_1AddV2MatMul_1:output:0add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
r
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*
_output_shapes

:
*
dtype0u
add_2AddV2MatMul_2:output:0add_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
p
MatMul_3BatchMatMulV2add:z:0	add_1:z:0*
T0*+
_output_shapes
:         

*
adj_y(H
Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 7
SqrtSqrtCast:y:0*
T0*
_output_shapes
: e
truedivRealDivMatMul_3:output:0Sqrt:y:0*
T0*+
_output_shapes
:         

U
SoftmaxSoftmaxtruediv:z:0*
T0*+
_output_shapes
:         

n
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes

:

*
dtype0o
MulMulSoftmax:softmax:0Mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         

c
MatMul_4BatchMatMulV2Mul:z:0	add_2:z:0*
T0*+
_output_shapes
:         
r
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes

:
*
dtype0u
add_3AddV2MatMul_4:output:0add_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
\
IdentityIdentity	add_3:z:0^NoOp*
T0*+
_output_shapes
:         
`

Identity_1Identitytruediv:z:0^NoOp*
T0*+
_output_shapes
:         

Б
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^Mul/ReadVariableOp^add/ReadVariableOp^add_1/ReadVariableOp^add_2/ReadVariableOp^add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         
: : : : : : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2,
add_3/ReadVariableOpadd_3/ReadVariableOp:P L
+
_output_shapes
:         


_user_specified_namex/0
ц
╔
B__inference_conv1d_layer_call_and_return_conditional_losses_318454

inputsA
+conv1d_expanddims_1_readvariableop_resource:
identityИв"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:м
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        j
IdentityIdentityConv1D/Squeeze:output:0^NoOp*
T0*+
_output_shapes
:         
k
NoOpNoOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         
: 2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╤
√
$__inference_signature_wrapper_317518
input_layer_1
unknown:


	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:

	unknown_6:

	unknown_7:

	unknown_8:


	unknown_9:
 

unknown_10:

unknown_11:


unknown_12:
identityИвStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinput_layer_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__wrapped_model_315819o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         d: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         d
'
_user_specified_nameinput_layer_1
й╬
░
&__inference_model_layer_call_fn_316003
input_layer_1H
2locally_connected1d_matmul_readvariableop_resource:

9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:M
;multi_level__block_attention_matmul_readvariableop_resource:O
=multi_level__block_attention_matmul_1_readvariableop_resource:O
=multi_level__block_attention_matmul_2_readvariableop_resource:J
8multi_level__block_attention_add_readvariableop_resource:
L
:multi_level__block_attention_add_1_readvariableop_resource:
L
:multi_level__block_attention_add_2_readvariableop_resource:
J
8multi_level__block_attention_mul_readvariableop_resource:

L
:multi_level__block_attention_add_3_readvariableop_resource:
H
2conv1d_conv1d_expanddims_1_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:
5
'dense_1_biasadd_readvariableop_resource:
identityИв)conv1d/Conv1D/ExpandDims_1/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/Tensordot/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв)locally_connected1d/MatMul/ReadVariableOpв2multi_level__block_attention/MatMul/ReadVariableOpв4multi_level__block_attention/MatMul_1/ReadVariableOpв4multi_level__block_attention/MatMul_2/ReadVariableOpв/multi_level__block_attention/Mul/ReadVariableOpв/multi_level__block_attention/add/ReadVariableOpв1multi_level__block_attention/add_1/ReadVariableOpв1multi_level__block_attention/add_2/ReadVariableOpв1multi_level__block_attention/add_3/ReadVariableOpД
zero_padding1d/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Д
zero_padding1d/PadPadinput_layer_1$zero_padding1d/Pad/paddings:output:0*
T0*+
_output_shapes
:         e|
'locally_connected1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            ~
)locally_connected1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       ~
)locally_connected1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ═
!locally_connected1d/strided_sliceStridedSlicezero_padding1d/Pad:output:00locally_connected1d/strided_slice/stack:output:02locally_connected1d/strided_slice/stack_1:output:02locally_connected1d/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskv
!locally_connected1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ┤
locally_connected1d/ReshapeReshape*locally_connected1d/strided_slice:output:0*locally_connected1d/Reshape/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       А
+locally_connected1d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_1StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_1/stack:output:04locally_connected1d/strided_slice_1/stack_1:output:04locally_connected1d/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_1Reshape,locally_connected1d/strided_slice_1:output:0,locally_connected1d/Reshape_1/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_2StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_2/stack:output:04locally_connected1d/strided_slice_2/stack_1:output:04locally_connected1d/strided_slice_2/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_2Reshape,locally_connected1d/strided_slice_2:output:0,locally_connected1d/Reshape_2/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    (       А
+locally_connected1d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_3StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_3/stack:output:04locally_connected1d/strided_slice_3/stack_1:output:04locally_connected1d/strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_3Reshape,locally_connected1d/strided_slice_3:output:0,locally_connected1d/Reshape_3/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"    (       А
+locally_connected1d/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    2       А
+locally_connected1d/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_4StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_4/stack:output:04locally_connected1d/strided_slice_4/stack_1:output:04locally_connected1d/strided_slice_4/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_4Reshape,locally_connected1d/strided_slice_4:output:0,locally_connected1d/Reshape_4/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*!
valueB"    2       А
+locally_connected1d/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    <       А
+locally_connected1d/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_5StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_5/stack:output:04locally_connected1d/strided_slice_5/stack_1:output:04locally_connected1d/strided_slice_5/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_5Reshape,locally_connected1d/strided_slice_5:output:0,locally_connected1d/Reshape_5/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"    <       А
+locally_connected1d/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    F       А
+locally_connected1d/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_6StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_6/stack:output:04locally_connected1d/strided_slice_6/stack_1:output:04locally_connected1d/strided_slice_6/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_6Reshape,locally_connected1d/strided_slice_6:output:0,locally_connected1d/Reshape_6/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*!
valueB"    F       А
+locally_connected1d/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    P       А
+locally_connected1d/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_7StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_7/stack:output:04locally_connected1d/strided_slice_7/stack_1:output:04locally_connected1d/strided_slice_7/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_7Reshape,locally_connected1d/strided_slice_7:output:0,locally_connected1d/Reshape_7/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*!
valueB"    P       А
+locally_connected1d/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    Z       А
+locally_connected1d/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_8StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_8/stack:output:04locally_connected1d/strided_slice_8/stack_1:output:04locally_connected1d/strided_slice_8/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_8Reshape,locally_connected1d/strided_slice_8:output:0,locally_connected1d/Reshape_8/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*!
valueB"    Z       А
+locally_connected1d/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    d       А
+locally_connected1d/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_9StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_9/stack:output:04locally_connected1d/strided_slice_9/stack_1:output:04locally_connected1d/strided_slice_9/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_9Reshape,locally_connected1d/strided_slice_9:output:0,locally_connected1d/Reshape_9/shape:output:0*
T0*+
_output_shapes
:         
a
locally_connected1d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Э
locally_connected1d/concatConcatV2$locally_connected1d/Reshape:output:0&locally_connected1d/Reshape_1:output:0&locally_connected1d/Reshape_2:output:0&locally_connected1d/Reshape_3:output:0&locally_connected1d/Reshape_4:output:0&locally_connected1d/Reshape_5:output:0&locally_connected1d/Reshape_6:output:0&locally_connected1d/Reshape_7:output:0&locally_connected1d/Reshape_8:output:0&locally_connected1d/Reshape_9:output:0(locally_connected1d/concat/axis:output:0*
N
*
T0*+
_output_shapes
:
         
а
)locally_connected1d/MatMul/ReadVariableOpReadVariableOp2locally_connected1d_matmul_readvariableop_resource*"
_output_shapes
:

*
dtype0╣
locally_connected1d/MatMulBatchMatMulV2#locally_connected1d/concat:output:01locally_connected1d/MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:
         l
locally_connected1d/ShapeShape#locally_connected1d/MatMul:output:0*
T0*
_output_shapes
:y
$locally_connected1d/Reshape_10/shapeConst*
_output_shapes
:*
dtype0*!
valueB"
          │
locally_connected1d/Reshape_10Reshape#locally_connected1d/MatMul:output:0-locally_connected1d/Reshape_10/shape:output:0*
T0*+
_output_shapes
:
         w
"locally_connected1d/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╢
locally_connected1d/transpose	Transpose'locally_connected1d/Reshape_10:output:0+locally_connected1d/transpose/perm:output:0*
T0*+
_output_shapes
:         
Ж
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       f
dense/Tensordot/ShapeShape!locally_connected1d/transpose:y:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╙
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ж
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ┤
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Л
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:а
dense/Tensordot/transpose	Transpose!locally_connected1d/transpose:y:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:         
Ь
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  Ь
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Х
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
о
2multi_level__block_attention/MatMul/ReadVariableOpReadVariableOp;multi_level__block_attention_matmul_readvariableop_resource*
_output_shapes

:*
dtype0╛
#multi_level__block_attention/MatMulBatchMatMulV2dense/BiasAdd:output:0:multi_level__block_attention/MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
▓
4multi_level__block_attention/MatMul_1/ReadVariableOpReadVariableOp=multi_level__block_attention_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0┬
%multi_level__block_attention/MatMul_1BatchMatMulV2dense/BiasAdd:output:0<multi_level__block_attention/MatMul_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
▓
4multi_level__block_attention/MatMul_2/ReadVariableOpReadVariableOp=multi_level__block_attention_matmul_2_readvariableop_resource*
_output_shapes

:*
dtype0┬
%multi_level__block_attention/MatMul_2BatchMatMulV2dense/BiasAdd:output:0<multi_level__block_attention/MatMul_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
и
/multi_level__block_attention/add/ReadVariableOpReadVariableOp8multi_level__block_attention_add_readvariableop_resource*
_output_shapes

:
*
dtype0╞
 multi_level__block_attention/addAddV2,multi_level__block_attention/MatMul:output:07multi_level__block_attention/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_1/ReadVariableOpReadVariableOp:multi_level__block_attention_add_1_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_1AddV2.multi_level__block_attention/MatMul_1:output:09multi_level__block_attention/add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_2/ReadVariableOpReadVariableOp:multi_level__block_attention_add_2_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_2AddV2.multi_level__block_attention/MatMul_2:output:09multi_level__block_attention/add_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
╟
%multi_level__block_attention/MatMul_3BatchMatMulV2$multi_level__block_attention/add:z:0&multi_level__block_attention/add_1:z:0*
T0*+
_output_shapes
:         

*
adj_y(e
#multi_level__block_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
З
!multi_level__block_attention/CastCast,multi_level__block_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: q
!multi_level__block_attention/SqrtSqrt%multi_level__block_attention/Cast:y:0*
T0*
_output_shapes
: ╝
$multi_level__block_attention/truedivRealDiv.multi_level__block_attention/MatMul_3:output:0%multi_level__block_attention/Sqrt:y:0*
T0*+
_output_shapes
:         

П
$multi_level__block_attention/SoftmaxSoftmax(multi_level__block_attention/truediv:z:0*
T0*+
_output_shapes
:         

и
/multi_level__block_attention/Mul/ReadVariableOpReadVariableOp8multi_level__block_attention_mul_readvariableop_resource*
_output_shapes

:

*
dtype0╞
 multi_level__block_attention/MulMul.multi_level__block_attention/Softmax:softmax:07multi_level__block_attention/Mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         

║
%multi_level__block_attention/MatMul_4BatchMatMulV2$multi_level__block_attention/Mul:z:0&multi_level__block_attention/add_2:z:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_3/ReadVariableOpReadVariableOp:multi_level__block_attention_add_3_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_3AddV2.multi_level__block_attention/MatMul_4:output:09multi_level__block_attention/add_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        п
conv1d/Conv1D/ExpandDims
ExpandDims&multi_level__block_attention/add_3:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
а
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╡
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:┴
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
О
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        s
activation/SigmoidSigmoidconv1d/Conv1D/Squeeze:output:0*
T0*+
_output_shapes
:         
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   |
flatten/ReshapeReshapeactivation/Sigmoid:y:0flatten/Const:output:0*
T0*'
_output_shapes
:         
q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :▒
global_average_pooling1d/MeanMeanconv1d/Conv1D/Squeeze:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         `
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Р
flatten_1/ReshapeReshape&global_average_pooling1d/Mean:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:         Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Л
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x
add/addAddV2flatten_1/Reshape:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentityadd/add:z:0^NoOp*
T0*'
_output_shapes
:         ┬
NoOpNoOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*^locally_connected1d/MatMul/ReadVariableOp3^multi_level__block_attention/MatMul/ReadVariableOp5^multi_level__block_attention/MatMul_1/ReadVariableOp5^multi_level__block_attention/MatMul_2/ReadVariableOp0^multi_level__block_attention/Mul/ReadVariableOp0^multi_level__block_attention/add/ReadVariableOp2^multi_level__block_attention/add_1/ReadVariableOp2^multi_level__block_attention/add_2/ReadVariableOp2^multi_level__block_attention/add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         d: : : : : : : : : : : : : : 2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2V
)locally_connected1d/MatMul/ReadVariableOp)locally_connected1d/MatMul/ReadVariableOp2h
2multi_level__block_attention/MatMul/ReadVariableOp2multi_level__block_attention/MatMul/ReadVariableOp2l
4multi_level__block_attention/MatMul_1/ReadVariableOp4multi_level__block_attention/MatMul_1/ReadVariableOp2l
4multi_level__block_attention/MatMul_2/ReadVariableOp4multi_level__block_attention/MatMul_2/ReadVariableOp2b
/multi_level__block_attention/Mul/ReadVariableOp/multi_level__block_attention/Mul/ReadVariableOp2b
/multi_level__block_attention/add/ReadVariableOp/multi_level__block_attention/add/ReadVariableOp2f
1multi_level__block_attention/add_1/ReadVariableOp1multi_level__block_attention/add_1/ReadVariableOp2f
1multi_level__block_attention/add_2/ReadVariableOp1multi_level__block_attention/add_2/ReadVariableOp2f
1multi_level__block_attention/add_3/ReadVariableOp1multi_level__block_attention/add_3/ReadVariableOp:Z V
+
_output_shapes
:         d
'
_user_specified_nameinput_layer_1
М
p
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_318476

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
У╬
й
&__inference_model_layer_call_fn_317673

inputsH
2locally_connected1d_matmul_readvariableop_resource:

9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:M
;multi_level__block_attention_matmul_readvariableop_resource:O
=multi_level__block_attention_matmul_1_readvariableop_resource:O
=multi_level__block_attention_matmul_2_readvariableop_resource:J
8multi_level__block_attention_add_readvariableop_resource:
L
:multi_level__block_attention_add_1_readvariableop_resource:
L
:multi_level__block_attention_add_2_readvariableop_resource:
J
8multi_level__block_attention_mul_readvariableop_resource:

L
:multi_level__block_attention_add_3_readvariableop_resource:
H
2conv1d_conv1d_expanddims_1_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:
5
'dense_1_biasadd_readvariableop_resource:
identityИв)conv1d/Conv1D/ExpandDims_1/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/Tensordot/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв)locally_connected1d/MatMul/ReadVariableOpв2multi_level__block_attention/MatMul/ReadVariableOpв4multi_level__block_attention/MatMul_1/ReadVariableOpв4multi_level__block_attention/MatMul_2/ReadVariableOpв/multi_level__block_attention/Mul/ReadVariableOpв/multi_level__block_attention/add/ReadVariableOpв1multi_level__block_attention/add_1/ReadVariableOpв1multi_level__block_attention/add_2/ReadVariableOpв1multi_level__block_attention/add_3/ReadVariableOpД
zero_padding1d/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       }
zero_padding1d/PadPadinputs$zero_padding1d/Pad/paddings:output:0*
T0*+
_output_shapes
:         e|
'locally_connected1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            ~
)locally_connected1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       ~
)locally_connected1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ═
!locally_connected1d/strided_sliceStridedSlicezero_padding1d/Pad:output:00locally_connected1d/strided_slice/stack:output:02locally_connected1d/strided_slice/stack_1:output:02locally_connected1d/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskv
!locally_connected1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ┤
locally_connected1d/ReshapeReshape*locally_connected1d/strided_slice:output:0*locally_connected1d/Reshape/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       А
+locally_connected1d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_1StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_1/stack:output:04locally_connected1d/strided_slice_1/stack_1:output:04locally_connected1d/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_1Reshape,locally_connected1d/strided_slice_1:output:0,locally_connected1d/Reshape_1/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_2StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_2/stack:output:04locally_connected1d/strided_slice_2/stack_1:output:04locally_connected1d/strided_slice_2/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_2Reshape,locally_connected1d/strided_slice_2:output:0,locally_connected1d/Reshape_2/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    (       А
+locally_connected1d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_3StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_3/stack:output:04locally_connected1d/strided_slice_3/stack_1:output:04locally_connected1d/strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_3Reshape,locally_connected1d/strided_slice_3:output:0,locally_connected1d/Reshape_3/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"    (       А
+locally_connected1d/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    2       А
+locally_connected1d/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_4StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_4/stack:output:04locally_connected1d/strided_slice_4/stack_1:output:04locally_connected1d/strided_slice_4/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_4Reshape,locally_connected1d/strided_slice_4:output:0,locally_connected1d/Reshape_4/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*!
valueB"    2       А
+locally_connected1d/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    <       А
+locally_connected1d/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_5StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_5/stack:output:04locally_connected1d/strided_slice_5/stack_1:output:04locally_connected1d/strided_slice_5/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_5Reshape,locally_connected1d/strided_slice_5:output:0,locally_connected1d/Reshape_5/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"    <       А
+locally_connected1d/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    F       А
+locally_connected1d/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_6StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_6/stack:output:04locally_connected1d/strided_slice_6/stack_1:output:04locally_connected1d/strided_slice_6/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_6Reshape,locally_connected1d/strided_slice_6:output:0,locally_connected1d/Reshape_6/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*!
valueB"    F       А
+locally_connected1d/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    P       А
+locally_connected1d/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_7StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_7/stack:output:04locally_connected1d/strided_slice_7/stack_1:output:04locally_connected1d/strided_slice_7/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_7Reshape,locally_connected1d/strided_slice_7:output:0,locally_connected1d/Reshape_7/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*!
valueB"    P       А
+locally_connected1d/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    Z       А
+locally_connected1d/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_8StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_8/stack:output:04locally_connected1d/strided_slice_8/stack_1:output:04locally_connected1d/strided_slice_8/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_8Reshape,locally_connected1d/strided_slice_8:output:0,locally_connected1d/Reshape_8/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*!
valueB"    Z       А
+locally_connected1d/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    d       А
+locally_connected1d/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_9StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_9/stack:output:04locally_connected1d/strided_slice_9/stack_1:output:04locally_connected1d/strided_slice_9/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_9Reshape,locally_connected1d/strided_slice_9:output:0,locally_connected1d/Reshape_9/shape:output:0*
T0*+
_output_shapes
:         
a
locally_connected1d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Э
locally_connected1d/concatConcatV2$locally_connected1d/Reshape:output:0&locally_connected1d/Reshape_1:output:0&locally_connected1d/Reshape_2:output:0&locally_connected1d/Reshape_3:output:0&locally_connected1d/Reshape_4:output:0&locally_connected1d/Reshape_5:output:0&locally_connected1d/Reshape_6:output:0&locally_connected1d/Reshape_7:output:0&locally_connected1d/Reshape_8:output:0&locally_connected1d/Reshape_9:output:0(locally_connected1d/concat/axis:output:0*
N
*
T0*+
_output_shapes
:
         
а
)locally_connected1d/MatMul/ReadVariableOpReadVariableOp2locally_connected1d_matmul_readvariableop_resource*"
_output_shapes
:

*
dtype0╣
locally_connected1d/MatMulBatchMatMulV2#locally_connected1d/concat:output:01locally_connected1d/MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:
         l
locally_connected1d/ShapeShape#locally_connected1d/MatMul:output:0*
T0*
_output_shapes
:y
$locally_connected1d/Reshape_10/shapeConst*
_output_shapes
:*
dtype0*!
valueB"
          │
locally_connected1d/Reshape_10Reshape#locally_connected1d/MatMul:output:0-locally_connected1d/Reshape_10/shape:output:0*
T0*+
_output_shapes
:
         w
"locally_connected1d/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╢
locally_connected1d/transpose	Transpose'locally_connected1d/Reshape_10:output:0+locally_connected1d/transpose/perm:output:0*
T0*+
_output_shapes
:         
Ж
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       f
dense/Tensordot/ShapeShape!locally_connected1d/transpose:y:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╙
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ж
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ┤
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Л
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:а
dense/Tensordot/transpose	Transpose!locally_connected1d/transpose:y:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:         
Ь
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  Ь
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Х
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
о
2multi_level__block_attention/MatMul/ReadVariableOpReadVariableOp;multi_level__block_attention_matmul_readvariableop_resource*
_output_shapes

:*
dtype0╛
#multi_level__block_attention/MatMulBatchMatMulV2dense/BiasAdd:output:0:multi_level__block_attention/MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
▓
4multi_level__block_attention/MatMul_1/ReadVariableOpReadVariableOp=multi_level__block_attention_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0┬
%multi_level__block_attention/MatMul_1BatchMatMulV2dense/BiasAdd:output:0<multi_level__block_attention/MatMul_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
▓
4multi_level__block_attention/MatMul_2/ReadVariableOpReadVariableOp=multi_level__block_attention_matmul_2_readvariableop_resource*
_output_shapes

:*
dtype0┬
%multi_level__block_attention/MatMul_2BatchMatMulV2dense/BiasAdd:output:0<multi_level__block_attention/MatMul_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
и
/multi_level__block_attention/add/ReadVariableOpReadVariableOp8multi_level__block_attention_add_readvariableop_resource*
_output_shapes

:
*
dtype0╞
 multi_level__block_attention/addAddV2,multi_level__block_attention/MatMul:output:07multi_level__block_attention/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_1/ReadVariableOpReadVariableOp:multi_level__block_attention_add_1_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_1AddV2.multi_level__block_attention/MatMul_1:output:09multi_level__block_attention/add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_2/ReadVariableOpReadVariableOp:multi_level__block_attention_add_2_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_2AddV2.multi_level__block_attention/MatMul_2:output:09multi_level__block_attention/add_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
╟
%multi_level__block_attention/MatMul_3BatchMatMulV2$multi_level__block_attention/add:z:0&multi_level__block_attention/add_1:z:0*
T0*+
_output_shapes
:         

*
adj_y(e
#multi_level__block_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
З
!multi_level__block_attention/CastCast,multi_level__block_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: q
!multi_level__block_attention/SqrtSqrt%multi_level__block_attention/Cast:y:0*
T0*
_output_shapes
: ╝
$multi_level__block_attention/truedivRealDiv.multi_level__block_attention/MatMul_3:output:0%multi_level__block_attention/Sqrt:y:0*
T0*+
_output_shapes
:         

П
$multi_level__block_attention/SoftmaxSoftmax(multi_level__block_attention/truediv:z:0*
T0*+
_output_shapes
:         

и
/multi_level__block_attention/Mul/ReadVariableOpReadVariableOp8multi_level__block_attention_mul_readvariableop_resource*
_output_shapes

:

*
dtype0╞
 multi_level__block_attention/MulMul.multi_level__block_attention/Softmax:softmax:07multi_level__block_attention/Mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         

║
%multi_level__block_attention/MatMul_4BatchMatMulV2$multi_level__block_attention/Mul:z:0&multi_level__block_attention/add_2:z:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_3/ReadVariableOpReadVariableOp:multi_level__block_attention_add_3_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_3AddV2.multi_level__block_attention/MatMul_4:output:09multi_level__block_attention/add_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        п
conv1d/Conv1D/ExpandDims
ExpandDims&multi_level__block_attention/add_3:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
а
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╡
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:┴
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
О
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        s
activation/SigmoidSigmoidconv1d/Conv1D/Squeeze:output:0*
T0*+
_output_shapes
:         
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   |
flatten/ReshapeReshapeactivation/Sigmoid:y:0flatten/Const:output:0*
T0*'
_output_shapes
:         
q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :▒
global_average_pooling1d/MeanMeanconv1d/Conv1D/Squeeze:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         `
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Р
flatten_1/ReshapeReshape&global_average_pooling1d/Mean:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:         Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Л
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x
add/addAddV2flatten_1/Reshape:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentityadd/add:z:0^NoOp*
T0*'
_output_shapes
:         ┬
NoOpNoOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*^locally_connected1d/MatMul/ReadVariableOp3^multi_level__block_attention/MatMul/ReadVariableOp5^multi_level__block_attention/MatMul_1/ReadVariableOp5^multi_level__block_attention/MatMul_2/ReadVariableOp0^multi_level__block_attention/Mul/ReadVariableOp0^multi_level__block_attention/add/ReadVariableOp2^multi_level__block_attention/add_1/ReadVariableOp2^multi_level__block_attention/add_2/ReadVariableOp2^multi_level__block_attention/add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         d: : : : : : : : : : : : : : 2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2V
)locally_connected1d/MatMul/ReadVariableOp)locally_connected1d/MatMul/ReadVariableOp2h
2multi_level__block_attention/MatMul/ReadVariableOp2multi_level__block_attention/MatMul/ReadVariableOp2l
4multi_level__block_attention/MatMul_1/ReadVariableOp4multi_level__block_attention/MatMul_1/ReadVariableOp2l
4multi_level__block_attention/MatMul_2/ReadVariableOp4multi_level__block_attention/MatMul_2/ReadVariableOp2b
/multi_level__block_attention/Mul/ReadVariableOp/multi_level__block_attention/Mul/ReadVariableOp2b
/multi_level__block_attention/add/ReadVariableOp/multi_level__block_attention/add/ReadVariableOp2f
1multi_level__block_attention/add_1/ReadVariableOp1multi_level__block_attention/add_1/ReadVariableOp2f
1multi_level__block_attention/add_2/ReadVariableOp1multi_level__block_attention/add_2/ReadVariableOp2f
1multi_level__block_attention/add_3/ReadVariableOp1multi_level__block_attention/add_3/ReadVariableOp:S O
+
_output_shapes
:         d
 
_user_specified_nameinputs
ХL
б
4__inference_locally_connected1d_layer_call_fn_318224

inputs4
matmul_readvariableop_resource:


identityИвMatMul/ReadVariableOph
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ш
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskb
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   x
ReshapeReshapestrided_slice:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:         
j
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       l
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ё
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskd
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ~
	Reshape_1Reshapestrided_slice_1:output:0Reshape_1/shape:output:0*
T0*+
_output_shapes
:         
j
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ё
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskd
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ~
	Reshape_2Reshapestrided_slice_2:output:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:         
j
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"           l
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    (       l
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ё
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskd
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ~
	Reshape_3Reshapestrided_slice_3:output:0Reshape_3/shape:output:0*
T0*+
_output_shapes
:         
j
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"    (       l
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    2       l
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ё
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskd
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ~
	Reshape_4Reshapestrided_slice_4:output:0Reshape_4/shape:output:0*
T0*+
_output_shapes
:         
j
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*!
valueB"    2       l
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    <       l
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ё
strided_slice_5StridedSliceinputsstrided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskd
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ~
	Reshape_5Reshapestrided_slice_5:output:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:         
j
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"    <       l
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    F       l
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ё
strided_slice_6StridedSliceinputsstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskd
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ~
	Reshape_6Reshapestrided_slice_6:output:0Reshape_6/shape:output:0*
T0*+
_output_shapes
:         
j
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*!
valueB"    F       l
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    P       l
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ё
strided_slice_7StridedSliceinputsstrided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskd
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ~
	Reshape_7Reshapestrided_slice_7:output:0Reshape_7/shape:output:0*
T0*+
_output_shapes
:         
j
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*!
valueB"    P       l
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    Z       l
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ё
strided_slice_8StridedSliceinputsstrided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskd
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ~
	Reshape_8Reshapestrided_slice_8:output:0Reshape_8/shape:output:0*
T0*+
_output_shapes
:         
j
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*!
valueB"    Z       l
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    d       l
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ё
strided_slice_9StridedSliceinputsstrided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskd
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ~
	Reshape_9Reshapestrided_slice_9:output:0Reshape_9/shape:output:0*
T0*+
_output_shapes
:         
M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : н
concatConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0concat/axis:output:0*
N
*
T0*+
_output_shapes
:
         
x
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*"
_output_shapes
:

*
dtype0}
MatMulBatchMatMulV2concat:output:0MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:
         D
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:e
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*!
valueB"
          w

Reshape_10ReshapeMatMul:output:0Reshape_10/shape:output:0*
T0*+
_output_shapes
:
         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          z
	transpose	TransposeReshape_10:output:0transpose/perm:output:0*
T0*+
_output_shapes
:         
`
IdentityIdentitytranspose:y:0^NoOp*
T0*+
_output_shapes
:         
^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         e: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:S O
+
_output_shapes
:         e
 
_user_specified_nameinputs
╛
G
+__inference_activation_layer_call_fn_318459

inputs
identityP
SigmoidSigmoidinputs*
T0*+
_output_shapes
:         
W
IdentityIdentitySigmoid:y:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
─╬
╦
A__inference_model_layer_call_and_return_conditional_losses_317477
input_layer_1H
2locally_connected1d_matmul_readvariableop_resource:

9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:M
;multi_level__block_attention_matmul_readvariableop_resource:O
=multi_level__block_attention_matmul_1_readvariableop_resource:O
=multi_level__block_attention_matmul_2_readvariableop_resource:J
8multi_level__block_attention_add_readvariableop_resource:
L
:multi_level__block_attention_add_1_readvariableop_resource:
L
:multi_level__block_attention_add_2_readvariableop_resource:
J
8multi_level__block_attention_mul_readvariableop_resource:

L
:multi_level__block_attention_add_3_readvariableop_resource:
H
2conv1d_conv1d_expanddims_1_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:
5
'dense_1_biasadd_readvariableop_resource:
identityИв)conv1d/Conv1D/ExpandDims_1/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/Tensordot/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв)locally_connected1d/MatMul/ReadVariableOpв2multi_level__block_attention/MatMul/ReadVariableOpв4multi_level__block_attention/MatMul_1/ReadVariableOpв4multi_level__block_attention/MatMul_2/ReadVariableOpв/multi_level__block_attention/Mul/ReadVariableOpв/multi_level__block_attention/add/ReadVariableOpв1multi_level__block_attention/add_1/ReadVariableOpв1multi_level__block_attention/add_2/ReadVariableOpв1multi_level__block_attention/add_3/ReadVariableOpД
zero_padding1d/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Д
zero_padding1d/PadPadinput_layer_1$zero_padding1d/Pad/paddings:output:0*
T0*+
_output_shapes
:         e|
'locally_connected1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            ~
)locally_connected1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       ~
)locally_connected1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ═
!locally_connected1d/strided_sliceStridedSlicezero_padding1d/Pad:output:00locally_connected1d/strided_slice/stack:output:02locally_connected1d/strided_slice/stack_1:output:02locally_connected1d/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskv
!locally_connected1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ┤
locally_connected1d/ReshapeReshape*locally_connected1d/strided_slice:output:0*locally_connected1d/Reshape/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       А
+locally_connected1d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_1StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_1/stack:output:04locally_connected1d/strided_slice_1/stack_1:output:04locally_connected1d/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_1Reshape,locally_connected1d/strided_slice_1:output:0,locally_connected1d/Reshape_1/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_2StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_2/stack:output:04locally_connected1d/strided_slice_2/stack_1:output:04locally_connected1d/strided_slice_2/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_2Reshape,locally_connected1d/strided_slice_2:output:0,locally_connected1d/Reshape_2/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    (       А
+locally_connected1d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_3StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_3/stack:output:04locally_connected1d/strided_slice_3/stack_1:output:04locally_connected1d/strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_3Reshape,locally_connected1d/strided_slice_3:output:0,locally_connected1d/Reshape_3/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"    (       А
+locally_connected1d/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    2       А
+locally_connected1d/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_4StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_4/stack:output:04locally_connected1d/strided_slice_4/stack_1:output:04locally_connected1d/strided_slice_4/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_4Reshape,locally_connected1d/strided_slice_4:output:0,locally_connected1d/Reshape_4/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*!
valueB"    2       А
+locally_connected1d/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    <       А
+locally_connected1d/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_5StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_5/stack:output:04locally_connected1d/strided_slice_5/stack_1:output:04locally_connected1d/strided_slice_5/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_5Reshape,locally_connected1d/strided_slice_5:output:0,locally_connected1d/Reshape_5/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"    <       А
+locally_connected1d/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    F       А
+locally_connected1d/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_6StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_6/stack:output:04locally_connected1d/strided_slice_6/stack_1:output:04locally_connected1d/strided_slice_6/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_6Reshape,locally_connected1d/strided_slice_6:output:0,locally_connected1d/Reshape_6/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*!
valueB"    F       А
+locally_connected1d/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    P       А
+locally_connected1d/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_7StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_7/stack:output:04locally_connected1d/strided_slice_7/stack_1:output:04locally_connected1d/strided_slice_7/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_7Reshape,locally_connected1d/strided_slice_7:output:0,locally_connected1d/Reshape_7/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*!
valueB"    P       А
+locally_connected1d/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    Z       А
+locally_connected1d/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_8StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_8/stack:output:04locally_connected1d/strided_slice_8/stack_1:output:04locally_connected1d/strided_slice_8/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_8Reshape,locally_connected1d/strided_slice_8:output:0,locally_connected1d/Reshape_8/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*!
valueB"    Z       А
+locally_connected1d/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    d       А
+locally_connected1d/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_9StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_9/stack:output:04locally_connected1d/strided_slice_9/stack_1:output:04locally_connected1d/strided_slice_9/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_9Reshape,locally_connected1d/strided_slice_9:output:0,locally_connected1d/Reshape_9/shape:output:0*
T0*+
_output_shapes
:         
a
locally_connected1d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Э
locally_connected1d/concatConcatV2$locally_connected1d/Reshape:output:0&locally_connected1d/Reshape_1:output:0&locally_connected1d/Reshape_2:output:0&locally_connected1d/Reshape_3:output:0&locally_connected1d/Reshape_4:output:0&locally_connected1d/Reshape_5:output:0&locally_connected1d/Reshape_6:output:0&locally_connected1d/Reshape_7:output:0&locally_connected1d/Reshape_8:output:0&locally_connected1d/Reshape_9:output:0(locally_connected1d/concat/axis:output:0*
N
*
T0*+
_output_shapes
:
         
а
)locally_connected1d/MatMul/ReadVariableOpReadVariableOp2locally_connected1d_matmul_readvariableop_resource*"
_output_shapes
:

*
dtype0╣
locally_connected1d/MatMulBatchMatMulV2#locally_connected1d/concat:output:01locally_connected1d/MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:
         l
locally_connected1d/ShapeShape#locally_connected1d/MatMul:output:0*
T0*
_output_shapes
:y
$locally_connected1d/Reshape_10/shapeConst*
_output_shapes
:*
dtype0*!
valueB"
          │
locally_connected1d/Reshape_10Reshape#locally_connected1d/MatMul:output:0-locally_connected1d/Reshape_10/shape:output:0*
T0*+
_output_shapes
:
         w
"locally_connected1d/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╢
locally_connected1d/transpose	Transpose'locally_connected1d/Reshape_10:output:0+locally_connected1d/transpose/perm:output:0*
T0*+
_output_shapes
:         
Ж
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       f
dense/Tensordot/ShapeShape!locally_connected1d/transpose:y:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╙
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ж
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ┤
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Л
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:а
dense/Tensordot/transpose	Transpose!locally_connected1d/transpose:y:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:         
Ь
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  Ь
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Х
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
о
2multi_level__block_attention/MatMul/ReadVariableOpReadVariableOp;multi_level__block_attention_matmul_readvariableop_resource*
_output_shapes

:*
dtype0╛
#multi_level__block_attention/MatMulBatchMatMulV2dense/BiasAdd:output:0:multi_level__block_attention/MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
▓
4multi_level__block_attention/MatMul_1/ReadVariableOpReadVariableOp=multi_level__block_attention_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0┬
%multi_level__block_attention/MatMul_1BatchMatMulV2dense/BiasAdd:output:0<multi_level__block_attention/MatMul_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
▓
4multi_level__block_attention/MatMul_2/ReadVariableOpReadVariableOp=multi_level__block_attention_matmul_2_readvariableop_resource*
_output_shapes

:*
dtype0┬
%multi_level__block_attention/MatMul_2BatchMatMulV2dense/BiasAdd:output:0<multi_level__block_attention/MatMul_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
и
/multi_level__block_attention/add/ReadVariableOpReadVariableOp8multi_level__block_attention_add_readvariableop_resource*
_output_shapes

:
*
dtype0╞
 multi_level__block_attention/addAddV2,multi_level__block_attention/MatMul:output:07multi_level__block_attention/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_1/ReadVariableOpReadVariableOp:multi_level__block_attention_add_1_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_1AddV2.multi_level__block_attention/MatMul_1:output:09multi_level__block_attention/add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_2/ReadVariableOpReadVariableOp:multi_level__block_attention_add_2_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_2AddV2.multi_level__block_attention/MatMul_2:output:09multi_level__block_attention/add_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
╟
%multi_level__block_attention/MatMul_3BatchMatMulV2$multi_level__block_attention/add:z:0&multi_level__block_attention/add_1:z:0*
T0*+
_output_shapes
:         

*
adj_y(e
#multi_level__block_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
З
!multi_level__block_attention/CastCast,multi_level__block_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: q
!multi_level__block_attention/SqrtSqrt%multi_level__block_attention/Cast:y:0*
T0*
_output_shapes
: ╝
$multi_level__block_attention/truedivRealDiv.multi_level__block_attention/MatMul_3:output:0%multi_level__block_attention/Sqrt:y:0*
T0*+
_output_shapes
:         

П
$multi_level__block_attention/SoftmaxSoftmax(multi_level__block_attention/truediv:z:0*
T0*+
_output_shapes
:         

и
/multi_level__block_attention/Mul/ReadVariableOpReadVariableOp8multi_level__block_attention_mul_readvariableop_resource*
_output_shapes

:

*
dtype0╞
 multi_level__block_attention/MulMul.multi_level__block_attention/Softmax:softmax:07multi_level__block_attention/Mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         

║
%multi_level__block_attention/MatMul_4BatchMatMulV2$multi_level__block_attention/Mul:z:0&multi_level__block_attention/add_2:z:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_3/ReadVariableOpReadVariableOp:multi_level__block_attention_add_3_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_3AddV2.multi_level__block_attention/MatMul_4:output:09multi_level__block_attention/add_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        п
conv1d/Conv1D/ExpandDims
ExpandDims&multi_level__block_attention/add_3:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
а
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╡
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:┴
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
О
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        s
activation/SigmoidSigmoidconv1d/Conv1D/Squeeze:output:0*
T0*+
_output_shapes
:         
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   |
flatten/ReshapeReshapeactivation/Sigmoid:y:0flatten/Const:output:0*
T0*'
_output_shapes
:         
q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :▒
global_average_pooling1d/MeanMeanconv1d/Conv1D/Squeeze:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         `
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Р
flatten_1/ReshapeReshape&global_average_pooling1d/Mean:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:         Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Л
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x
add/addAddV2flatten_1/Reshape:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentityadd/add:z:0^NoOp*
T0*'
_output_shapes
:         ┬
NoOpNoOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*^locally_connected1d/MatMul/ReadVariableOp3^multi_level__block_attention/MatMul/ReadVariableOp5^multi_level__block_attention/MatMul_1/ReadVariableOp5^multi_level__block_attention/MatMul_2/ReadVariableOp0^multi_level__block_attention/Mul/ReadVariableOp0^multi_level__block_attention/add/ReadVariableOp2^multi_level__block_attention/add_1/ReadVariableOp2^multi_level__block_attention/add_2/ReadVariableOp2^multi_level__block_attention/add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         d: : : : : : : : : : : : : : 2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2V
)locally_connected1d/MatMul/ReadVariableOp)locally_connected1d/MatMul/ReadVariableOp2h
2multi_level__block_attention/MatMul/ReadVariableOp2multi_level__block_attention/MatMul/ReadVariableOp2l
4multi_level__block_attention/MatMul_1/ReadVariableOp4multi_level__block_attention/MatMul_1/ReadVariableOp2l
4multi_level__block_attention/MatMul_2/ReadVariableOp4multi_level__block_attention/MatMul_2/ReadVariableOp2b
/multi_level__block_attention/Mul/ReadVariableOp/multi_level__block_attention/Mul/ReadVariableOp2b
/multi_level__block_attention/add/ReadVariableOp/multi_level__block_attention/add/ReadVariableOp2f
1multi_level__block_attention/add_1/ReadVariableOp1multi_level__block_attention/add_1/ReadVariableOp2f
1multi_level__block_attention/add_2/ReadVariableOp1multi_level__block_attention/add_2/ReadVariableOp2f
1multi_level__block_attention/add_3/ReadVariableOp1multi_level__block_attention/add_3/ReadVariableOp:Z V
+
_output_shapes
:         d
'
_user_specified_nameinput_layer_1
а
D
(__inference_flatten_layer_call_fn_318482

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         
X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
н
▌
&__inference_dense_layer_call_fn_318328

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:         
К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         
z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
У╬
й
&__inference_model_layer_call_fn_317828

inputsH
2locally_connected1d_matmul_readvariableop_resource:

9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:M
;multi_level__block_attention_matmul_readvariableop_resource:O
=multi_level__block_attention_matmul_1_readvariableop_resource:O
=multi_level__block_attention_matmul_2_readvariableop_resource:J
8multi_level__block_attention_add_readvariableop_resource:
L
:multi_level__block_attention_add_1_readvariableop_resource:
L
:multi_level__block_attention_add_2_readvariableop_resource:
J
8multi_level__block_attention_mul_readvariableop_resource:

L
:multi_level__block_attention_add_3_readvariableop_resource:
H
2conv1d_conv1d_expanddims_1_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:
5
'dense_1_biasadd_readvariableop_resource:
identityИв)conv1d/Conv1D/ExpandDims_1/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/Tensordot/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв)locally_connected1d/MatMul/ReadVariableOpв2multi_level__block_attention/MatMul/ReadVariableOpв4multi_level__block_attention/MatMul_1/ReadVariableOpв4multi_level__block_attention/MatMul_2/ReadVariableOpв/multi_level__block_attention/Mul/ReadVariableOpв/multi_level__block_attention/add/ReadVariableOpв1multi_level__block_attention/add_1/ReadVariableOpв1multi_level__block_attention/add_2/ReadVariableOpв1multi_level__block_attention/add_3/ReadVariableOpД
zero_padding1d/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       }
zero_padding1d/PadPadinputs$zero_padding1d/Pad/paddings:output:0*
T0*+
_output_shapes
:         e|
'locally_connected1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            ~
)locally_connected1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       ~
)locally_connected1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ═
!locally_connected1d/strided_sliceStridedSlicezero_padding1d/Pad:output:00locally_connected1d/strided_slice/stack:output:02locally_connected1d/strided_slice/stack_1:output:02locally_connected1d/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskv
!locally_connected1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ┤
locally_connected1d/ReshapeReshape*locally_connected1d/strided_slice:output:0*locally_connected1d/Reshape/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       А
+locally_connected1d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_1StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_1/stack:output:04locally_connected1d/strided_slice_1/stack_1:output:04locally_connected1d/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_1Reshape,locally_connected1d/strided_slice_1:output:0,locally_connected1d/Reshape_1/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_2StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_2/stack:output:04locally_connected1d/strided_slice_2/stack_1:output:04locally_connected1d/strided_slice_2/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_2Reshape,locally_connected1d/strided_slice_2:output:0,locally_connected1d/Reshape_2/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    (       А
+locally_connected1d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_3StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_3/stack:output:04locally_connected1d/strided_slice_3/stack_1:output:04locally_connected1d/strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_3Reshape,locally_connected1d/strided_slice_3:output:0,locally_connected1d/Reshape_3/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"    (       А
+locally_connected1d/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    2       А
+locally_connected1d/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_4StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_4/stack:output:04locally_connected1d/strided_slice_4/stack_1:output:04locally_connected1d/strided_slice_4/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_4Reshape,locally_connected1d/strided_slice_4:output:0,locally_connected1d/Reshape_4/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*!
valueB"    2       А
+locally_connected1d/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    <       А
+locally_connected1d/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_5StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_5/stack:output:04locally_connected1d/strided_slice_5/stack_1:output:04locally_connected1d/strided_slice_5/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_5Reshape,locally_connected1d/strided_slice_5:output:0,locally_connected1d/Reshape_5/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"    <       А
+locally_connected1d/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    F       А
+locally_connected1d/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_6StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_6/stack:output:04locally_connected1d/strided_slice_6/stack_1:output:04locally_connected1d/strided_slice_6/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_6Reshape,locally_connected1d/strided_slice_6:output:0,locally_connected1d/Reshape_6/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*!
valueB"    F       А
+locally_connected1d/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    P       А
+locally_connected1d/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_7StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_7/stack:output:04locally_connected1d/strided_slice_7/stack_1:output:04locally_connected1d/strided_slice_7/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_7Reshape,locally_connected1d/strided_slice_7:output:0,locally_connected1d/Reshape_7/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*!
valueB"    P       А
+locally_connected1d/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    Z       А
+locally_connected1d/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_8StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_8/stack:output:04locally_connected1d/strided_slice_8/stack_1:output:04locally_connected1d/strided_slice_8/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_8Reshape,locally_connected1d/strided_slice_8:output:0,locally_connected1d/Reshape_8/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*!
valueB"    Z       А
+locally_connected1d/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    d       А
+locally_connected1d/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_9StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_9/stack:output:04locally_connected1d/strided_slice_9/stack_1:output:04locally_connected1d/strided_slice_9/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_9Reshape,locally_connected1d/strided_slice_9:output:0,locally_connected1d/Reshape_9/shape:output:0*
T0*+
_output_shapes
:         
a
locally_connected1d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Э
locally_connected1d/concatConcatV2$locally_connected1d/Reshape:output:0&locally_connected1d/Reshape_1:output:0&locally_connected1d/Reshape_2:output:0&locally_connected1d/Reshape_3:output:0&locally_connected1d/Reshape_4:output:0&locally_connected1d/Reshape_5:output:0&locally_connected1d/Reshape_6:output:0&locally_connected1d/Reshape_7:output:0&locally_connected1d/Reshape_8:output:0&locally_connected1d/Reshape_9:output:0(locally_connected1d/concat/axis:output:0*
N
*
T0*+
_output_shapes
:
         
а
)locally_connected1d/MatMul/ReadVariableOpReadVariableOp2locally_connected1d_matmul_readvariableop_resource*"
_output_shapes
:

*
dtype0╣
locally_connected1d/MatMulBatchMatMulV2#locally_connected1d/concat:output:01locally_connected1d/MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:
         l
locally_connected1d/ShapeShape#locally_connected1d/MatMul:output:0*
T0*
_output_shapes
:y
$locally_connected1d/Reshape_10/shapeConst*
_output_shapes
:*
dtype0*!
valueB"
          │
locally_connected1d/Reshape_10Reshape#locally_connected1d/MatMul:output:0-locally_connected1d/Reshape_10/shape:output:0*
T0*+
_output_shapes
:
         w
"locally_connected1d/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╢
locally_connected1d/transpose	Transpose'locally_connected1d/Reshape_10:output:0+locally_connected1d/transpose/perm:output:0*
T0*+
_output_shapes
:         
Ж
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       f
dense/Tensordot/ShapeShape!locally_connected1d/transpose:y:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╙
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ж
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ┤
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Л
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:а
dense/Tensordot/transpose	Transpose!locally_connected1d/transpose:y:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:         
Ь
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  Ь
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Х
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
о
2multi_level__block_attention/MatMul/ReadVariableOpReadVariableOp;multi_level__block_attention_matmul_readvariableop_resource*
_output_shapes

:*
dtype0╛
#multi_level__block_attention/MatMulBatchMatMulV2dense/BiasAdd:output:0:multi_level__block_attention/MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
▓
4multi_level__block_attention/MatMul_1/ReadVariableOpReadVariableOp=multi_level__block_attention_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0┬
%multi_level__block_attention/MatMul_1BatchMatMulV2dense/BiasAdd:output:0<multi_level__block_attention/MatMul_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
▓
4multi_level__block_attention/MatMul_2/ReadVariableOpReadVariableOp=multi_level__block_attention_matmul_2_readvariableop_resource*
_output_shapes

:*
dtype0┬
%multi_level__block_attention/MatMul_2BatchMatMulV2dense/BiasAdd:output:0<multi_level__block_attention/MatMul_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
и
/multi_level__block_attention/add/ReadVariableOpReadVariableOp8multi_level__block_attention_add_readvariableop_resource*
_output_shapes

:
*
dtype0╞
 multi_level__block_attention/addAddV2,multi_level__block_attention/MatMul:output:07multi_level__block_attention/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_1/ReadVariableOpReadVariableOp:multi_level__block_attention_add_1_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_1AddV2.multi_level__block_attention/MatMul_1:output:09multi_level__block_attention/add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_2/ReadVariableOpReadVariableOp:multi_level__block_attention_add_2_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_2AddV2.multi_level__block_attention/MatMul_2:output:09multi_level__block_attention/add_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
╟
%multi_level__block_attention/MatMul_3BatchMatMulV2$multi_level__block_attention/add:z:0&multi_level__block_attention/add_1:z:0*
T0*+
_output_shapes
:         

*
adj_y(e
#multi_level__block_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
З
!multi_level__block_attention/CastCast,multi_level__block_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: q
!multi_level__block_attention/SqrtSqrt%multi_level__block_attention/Cast:y:0*
T0*
_output_shapes
: ╝
$multi_level__block_attention/truedivRealDiv.multi_level__block_attention/MatMul_3:output:0%multi_level__block_attention/Sqrt:y:0*
T0*+
_output_shapes
:         

П
$multi_level__block_attention/SoftmaxSoftmax(multi_level__block_attention/truediv:z:0*
T0*+
_output_shapes
:         

и
/multi_level__block_attention/Mul/ReadVariableOpReadVariableOp8multi_level__block_attention_mul_readvariableop_resource*
_output_shapes

:

*
dtype0╞
 multi_level__block_attention/MulMul.multi_level__block_attention/Softmax:softmax:07multi_level__block_attention/Mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         

║
%multi_level__block_attention/MatMul_4BatchMatMulV2$multi_level__block_attention/Mul:z:0&multi_level__block_attention/add_2:z:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_3/ReadVariableOpReadVariableOp:multi_level__block_attention_add_3_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_3AddV2.multi_level__block_attention/MatMul_4:output:09multi_level__block_attention/add_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        п
conv1d/Conv1D/ExpandDims
ExpandDims&multi_level__block_attention/add_3:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
а
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╡
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:┴
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
О
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        s
activation/SigmoidSigmoidconv1d/Conv1D/Squeeze:output:0*
T0*+
_output_shapes
:         
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   |
flatten/ReshapeReshapeactivation/Sigmoid:y:0flatten/Const:output:0*
T0*'
_output_shapes
:         
q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :▒
global_average_pooling1d/MeanMeanconv1d/Conv1D/Squeeze:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         `
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Р
flatten_1/ReshapeReshape&global_average_pooling1d/Mean:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:         Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Л
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x
add/addAddV2flatten_1/Reshape:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentityadd/add:z:0^NoOp*
T0*'
_output_shapes
:         ┬
NoOpNoOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*^locally_connected1d/MatMul/ReadVariableOp3^multi_level__block_attention/MatMul/ReadVariableOp5^multi_level__block_attention/MatMul_1/ReadVariableOp5^multi_level__block_attention/MatMul_2/ReadVariableOp0^multi_level__block_attention/Mul/ReadVariableOp0^multi_level__block_attention/add/ReadVariableOp2^multi_level__block_attention/add_1/ReadVariableOp2^multi_level__block_attention/add_2/ReadVariableOp2^multi_level__block_attention/add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         d: : : : : : : : : : : : : : 2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2V
)locally_connected1d/MatMul/ReadVariableOp)locally_connected1d/MatMul/ReadVariableOp2h
2multi_level__block_attention/MatMul/ReadVariableOp2multi_level__block_attention/MatMul/ReadVariableOp2l
4multi_level__block_attention/MatMul_1/ReadVariableOp4multi_level__block_attention/MatMul_1/ReadVariableOp2l
4multi_level__block_attention/MatMul_2/ReadVariableOp4multi_level__block_attention/MatMul_2/ReadVariableOp2b
/multi_level__block_attention/Mul/ReadVariableOp/multi_level__block_attention/Mul/ReadVariableOp2b
/multi_level__block_attention/add/ReadVariableOp/multi_level__block_attention/add/ReadVariableOp2f
1multi_level__block_attention/add_1/ReadVariableOp1multi_level__block_attention/add_1/ReadVariableOp2f
1multi_level__block_attention/add_2/ReadVariableOp1multi_level__block_attention/add_2/ReadVariableOp2f
1multi_level__block_attention/add_3/ReadVariableOp1multi_level__block_attention/add_3/ReadVariableOp:S O
+
_output_shapes
:         d
 
_user_specified_nameinputs
╞	
Ї
C__inference_dense_1_layer_call_and_return_conditional_losses_318520

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
╝
k
?__inference_add_layer_call_and_return_conditional_losses_318532
inputs_0
inputs_1
identityR
addAddV2inputs_0inputs_1*
T0*'
_output_shapes
:         O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         :Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
┘
b
F__inference_activation_layer_call_and_return_conditional_losses_318464

inputs
identityP
SigmoidSigmoidinputs*
T0*+
_output_shapes
:         
W
IdentityIdentitySigmoid:y:0*
T0*+
_output_shapes
:         
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
л	
┘
(__inference_dense_1_layer_call_fn_318510

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         

 
_user_specified_nameinputs
й╬
░
&__inference_model_layer_call_fn_317167
input_layer_1H
2locally_connected1d_matmul_readvariableop_resource:

9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:M
;multi_level__block_attention_matmul_readvariableop_resource:O
=multi_level__block_attention_matmul_1_readvariableop_resource:O
=multi_level__block_attention_matmul_2_readvariableop_resource:J
8multi_level__block_attention_add_readvariableop_resource:
L
:multi_level__block_attention_add_1_readvariableop_resource:
L
:multi_level__block_attention_add_2_readvariableop_resource:
J
8multi_level__block_attention_mul_readvariableop_resource:

L
:multi_level__block_attention_add_3_readvariableop_resource:
H
2conv1d_conv1d_expanddims_1_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:
5
'dense_1_biasadd_readvariableop_resource:
identityИв)conv1d/Conv1D/ExpandDims_1/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/Tensordot/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв)locally_connected1d/MatMul/ReadVariableOpв2multi_level__block_attention/MatMul/ReadVariableOpв4multi_level__block_attention/MatMul_1/ReadVariableOpв4multi_level__block_attention/MatMul_2/ReadVariableOpв/multi_level__block_attention/Mul/ReadVariableOpв/multi_level__block_attention/add/ReadVariableOpв1multi_level__block_attention/add_1/ReadVariableOpв1multi_level__block_attention/add_2/ReadVariableOpв1multi_level__block_attention/add_3/ReadVariableOpД
zero_padding1d/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       Д
zero_padding1d/PadPadinput_layer_1$zero_padding1d/Pad/paddings:output:0*
T0*+
_output_shapes
:         e|
'locally_connected1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            ~
)locally_connected1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       ~
)locally_connected1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ═
!locally_connected1d/strided_sliceStridedSlicezero_padding1d/Pad:output:00locally_connected1d/strided_slice/stack:output:02locally_connected1d/strided_slice/stack_1:output:02locally_connected1d/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskv
!locally_connected1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ┤
locally_connected1d/ReshapeReshape*locally_connected1d/strided_slice:output:0*locally_connected1d/Reshape/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       А
+locally_connected1d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_1StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_1/stack:output:04locally_connected1d/strided_slice_1/stack_1:output:04locally_connected1d/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_1Reshape,locally_connected1d/strided_slice_1:output:0,locally_connected1d/Reshape_1/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_2StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_2/stack:output:04locally_connected1d/strided_slice_2/stack_1:output:04locally_connected1d/strided_slice_2/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_2Reshape,locally_connected1d/strided_slice_2:output:0,locally_connected1d/Reshape_2/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    (       А
+locally_connected1d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_3StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_3/stack:output:04locally_connected1d/strided_slice_3/stack_1:output:04locally_connected1d/strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_3Reshape,locally_connected1d/strided_slice_3:output:0,locally_connected1d/Reshape_3/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"    (       А
+locally_connected1d/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    2       А
+locally_connected1d/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_4StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_4/stack:output:04locally_connected1d/strided_slice_4/stack_1:output:04locally_connected1d/strided_slice_4/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_4Reshape,locally_connected1d/strided_slice_4:output:0,locally_connected1d/Reshape_4/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*!
valueB"    2       А
+locally_connected1d/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    <       А
+locally_connected1d/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_5StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_5/stack:output:04locally_connected1d/strided_slice_5/stack_1:output:04locally_connected1d/strided_slice_5/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_5Reshape,locally_connected1d/strided_slice_5:output:0,locally_connected1d/Reshape_5/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"    <       А
+locally_connected1d/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    F       А
+locally_connected1d/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_6StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_6/stack:output:04locally_connected1d/strided_slice_6/stack_1:output:04locally_connected1d/strided_slice_6/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_6Reshape,locally_connected1d/strided_slice_6:output:0,locally_connected1d/Reshape_6/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*!
valueB"    F       А
+locally_connected1d/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    P       А
+locally_connected1d/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_7StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_7/stack:output:04locally_connected1d/strided_slice_7/stack_1:output:04locally_connected1d/strided_slice_7/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_7Reshape,locally_connected1d/strided_slice_7:output:0,locally_connected1d/Reshape_7/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*!
valueB"    P       А
+locally_connected1d/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    Z       А
+locally_connected1d/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_8StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_8/stack:output:04locally_connected1d/strided_slice_8/stack_1:output:04locally_connected1d/strided_slice_8/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_8Reshape,locally_connected1d/strided_slice_8:output:0,locally_connected1d/Reshape_8/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*!
valueB"    Z       А
+locally_connected1d/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    d       А
+locally_connected1d/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_9StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_9/stack:output:04locally_connected1d/strided_slice_9/stack_1:output:04locally_connected1d/strided_slice_9/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_9Reshape,locally_connected1d/strided_slice_9:output:0,locally_connected1d/Reshape_9/shape:output:0*
T0*+
_output_shapes
:         
a
locally_connected1d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Э
locally_connected1d/concatConcatV2$locally_connected1d/Reshape:output:0&locally_connected1d/Reshape_1:output:0&locally_connected1d/Reshape_2:output:0&locally_connected1d/Reshape_3:output:0&locally_connected1d/Reshape_4:output:0&locally_connected1d/Reshape_5:output:0&locally_connected1d/Reshape_6:output:0&locally_connected1d/Reshape_7:output:0&locally_connected1d/Reshape_8:output:0&locally_connected1d/Reshape_9:output:0(locally_connected1d/concat/axis:output:0*
N
*
T0*+
_output_shapes
:
         
а
)locally_connected1d/MatMul/ReadVariableOpReadVariableOp2locally_connected1d_matmul_readvariableop_resource*"
_output_shapes
:

*
dtype0╣
locally_connected1d/MatMulBatchMatMulV2#locally_connected1d/concat:output:01locally_connected1d/MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:
         l
locally_connected1d/ShapeShape#locally_connected1d/MatMul:output:0*
T0*
_output_shapes
:y
$locally_connected1d/Reshape_10/shapeConst*
_output_shapes
:*
dtype0*!
valueB"
          │
locally_connected1d/Reshape_10Reshape#locally_connected1d/MatMul:output:0-locally_connected1d/Reshape_10/shape:output:0*
T0*+
_output_shapes
:
         w
"locally_connected1d/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╢
locally_connected1d/transpose	Transpose'locally_connected1d/Reshape_10:output:0+locally_connected1d/transpose/perm:output:0*
T0*+
_output_shapes
:         
Ж
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       f
dense/Tensordot/ShapeShape!locally_connected1d/transpose:y:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╙
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ж
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ┤
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Л
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:а
dense/Tensordot/transpose	Transpose!locally_connected1d/transpose:y:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:         
Ь
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  Ь
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Х
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
о
2multi_level__block_attention/MatMul/ReadVariableOpReadVariableOp;multi_level__block_attention_matmul_readvariableop_resource*
_output_shapes

:*
dtype0╛
#multi_level__block_attention/MatMulBatchMatMulV2dense/BiasAdd:output:0:multi_level__block_attention/MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
▓
4multi_level__block_attention/MatMul_1/ReadVariableOpReadVariableOp=multi_level__block_attention_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0┬
%multi_level__block_attention/MatMul_1BatchMatMulV2dense/BiasAdd:output:0<multi_level__block_attention/MatMul_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
▓
4multi_level__block_attention/MatMul_2/ReadVariableOpReadVariableOp=multi_level__block_attention_matmul_2_readvariableop_resource*
_output_shapes

:*
dtype0┬
%multi_level__block_attention/MatMul_2BatchMatMulV2dense/BiasAdd:output:0<multi_level__block_attention/MatMul_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
и
/multi_level__block_attention/add/ReadVariableOpReadVariableOp8multi_level__block_attention_add_readvariableop_resource*
_output_shapes

:
*
dtype0╞
 multi_level__block_attention/addAddV2,multi_level__block_attention/MatMul:output:07multi_level__block_attention/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_1/ReadVariableOpReadVariableOp:multi_level__block_attention_add_1_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_1AddV2.multi_level__block_attention/MatMul_1:output:09multi_level__block_attention/add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_2/ReadVariableOpReadVariableOp:multi_level__block_attention_add_2_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_2AddV2.multi_level__block_attention/MatMul_2:output:09multi_level__block_attention/add_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
╟
%multi_level__block_attention/MatMul_3BatchMatMulV2$multi_level__block_attention/add:z:0&multi_level__block_attention/add_1:z:0*
T0*+
_output_shapes
:         

*
adj_y(e
#multi_level__block_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
З
!multi_level__block_attention/CastCast,multi_level__block_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: q
!multi_level__block_attention/SqrtSqrt%multi_level__block_attention/Cast:y:0*
T0*
_output_shapes
: ╝
$multi_level__block_attention/truedivRealDiv.multi_level__block_attention/MatMul_3:output:0%multi_level__block_attention/Sqrt:y:0*
T0*+
_output_shapes
:         

П
$multi_level__block_attention/SoftmaxSoftmax(multi_level__block_attention/truediv:z:0*
T0*+
_output_shapes
:         

и
/multi_level__block_attention/Mul/ReadVariableOpReadVariableOp8multi_level__block_attention_mul_readvariableop_resource*
_output_shapes

:

*
dtype0╞
 multi_level__block_attention/MulMul.multi_level__block_attention/Softmax:softmax:07multi_level__block_attention/Mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         

║
%multi_level__block_attention/MatMul_4BatchMatMulV2$multi_level__block_attention/Mul:z:0&multi_level__block_attention/add_2:z:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_3/ReadVariableOpReadVariableOp:multi_level__block_attention_add_3_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_3AddV2.multi_level__block_attention/MatMul_4:output:09multi_level__block_attention/add_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        п
conv1d/Conv1D/ExpandDims
ExpandDims&multi_level__block_attention/add_3:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
а
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╡
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:┴
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
О
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        s
activation/SigmoidSigmoidconv1d/Conv1D/Squeeze:output:0*
T0*+
_output_shapes
:         
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   |
flatten/ReshapeReshapeactivation/Sigmoid:y:0flatten/Const:output:0*
T0*'
_output_shapes
:         
q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :▒
global_average_pooling1d/MeanMeanconv1d/Conv1D/Squeeze:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         `
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Р
flatten_1/ReshapeReshape&global_average_pooling1d/Mean:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:         Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Л
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x
add/addAddV2flatten_1/Reshape:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentityadd/add:z:0^NoOp*
T0*'
_output_shapes
:         ┬
NoOpNoOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*^locally_connected1d/MatMul/ReadVariableOp3^multi_level__block_attention/MatMul/ReadVariableOp5^multi_level__block_attention/MatMul_1/ReadVariableOp5^multi_level__block_attention/MatMul_2/ReadVariableOp0^multi_level__block_attention/Mul/ReadVariableOp0^multi_level__block_attention/add/ReadVariableOp2^multi_level__block_attention/add_1/ReadVariableOp2^multi_level__block_attention/add_2/ReadVariableOp2^multi_level__block_attention/add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         d: : : : : : : : : : : : : : 2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2V
)locally_connected1d/MatMul/ReadVariableOp)locally_connected1d/MatMul/ReadVariableOp2h
2multi_level__block_attention/MatMul/ReadVariableOp2multi_level__block_attention/MatMul/ReadVariableOp2l
4multi_level__block_attention/MatMul_1/ReadVariableOp4multi_level__block_attention/MatMul_1/ReadVariableOp2l
4multi_level__block_attention/MatMul_2/ReadVariableOp4multi_level__block_attention/MatMul_2/ReadVariableOp2b
/multi_level__block_attention/Mul/ReadVariableOp/multi_level__block_attention/Mul/ReadVariableOp2b
/multi_level__block_attention/add/ReadVariableOp/multi_level__block_attention/add/ReadVariableOp2f
1multi_level__block_attention/add_1/ReadVariableOp1multi_level__block_attention/add_1/ReadVariableOp2f
1multi_level__block_attention/add_2/ReadVariableOp1multi_level__block_attention/add_2/ReadVariableOp2f
1multi_level__block_attention/add_3/ReadVariableOp1multi_level__block_attention/add_3/ReadVariableOp:Z V
+
_output_shapes
:         d
'
_user_specified_nameinput_layer_1
о╬
─
A__inference_model_layer_call_and_return_conditional_losses_318138

inputsH
2locally_connected1d_matmul_readvariableop_resource:

9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:M
;multi_level__block_attention_matmul_readvariableop_resource:O
=multi_level__block_attention_matmul_1_readvariableop_resource:O
=multi_level__block_attention_matmul_2_readvariableop_resource:J
8multi_level__block_attention_add_readvariableop_resource:
L
:multi_level__block_attention_add_1_readvariableop_resource:
L
:multi_level__block_attention_add_2_readvariableop_resource:
J
8multi_level__block_attention_mul_readvariableop_resource:

L
:multi_level__block_attention_add_3_readvariableop_resource:
H
2conv1d_conv1d_expanddims_1_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:
5
'dense_1_biasadd_readvariableop_resource:
identityИв)conv1d/Conv1D/ExpandDims_1/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/Tensordot/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв)locally_connected1d/MatMul/ReadVariableOpв2multi_level__block_attention/MatMul/ReadVariableOpв4multi_level__block_attention/MatMul_1/ReadVariableOpв4multi_level__block_attention/MatMul_2/ReadVariableOpв/multi_level__block_attention/Mul/ReadVariableOpв/multi_level__block_attention/add/ReadVariableOpв1multi_level__block_attention/add_1/ReadVariableOpв1multi_level__block_attention/add_2/ReadVariableOpв1multi_level__block_attention/add_3/ReadVariableOpД
zero_padding1d/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       }
zero_padding1d/PadPadinputs$zero_padding1d/Pad/paddings:output:0*
T0*+
_output_shapes
:         e|
'locally_connected1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            ~
)locally_connected1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       ~
)locally_connected1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ═
!locally_connected1d/strided_sliceStridedSlicezero_padding1d/Pad:output:00locally_connected1d/strided_slice/stack:output:02locally_connected1d/strided_slice/stack_1:output:02locally_connected1d/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskv
!locally_connected1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ┤
locally_connected1d/ReshapeReshape*locally_connected1d/strided_slice:output:0*locally_connected1d/Reshape/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       А
+locally_connected1d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_1StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_1/stack:output:04locally_connected1d/strided_slice_1/stack_1:output:04locally_connected1d/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_1Reshape,locally_connected1d/strided_slice_1:output:0,locally_connected1d/Reshape_1/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_2StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_2/stack:output:04locally_connected1d/strided_slice_2/stack_1:output:04locally_connected1d/strided_slice_2/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_2Reshape,locally_connected1d/strided_slice_2:output:0,locally_connected1d/Reshape_2/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    (       А
+locally_connected1d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_3StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_3/stack:output:04locally_connected1d/strided_slice_3/stack_1:output:04locally_connected1d/strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_3Reshape,locally_connected1d/strided_slice_3:output:0,locally_connected1d/Reshape_3/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"    (       А
+locally_connected1d/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    2       А
+locally_connected1d/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_4StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_4/stack:output:04locally_connected1d/strided_slice_4/stack_1:output:04locally_connected1d/strided_slice_4/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_4Reshape,locally_connected1d/strided_slice_4:output:0,locally_connected1d/Reshape_4/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*!
valueB"    2       А
+locally_connected1d/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    <       А
+locally_connected1d/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_5StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_5/stack:output:04locally_connected1d/strided_slice_5/stack_1:output:04locally_connected1d/strided_slice_5/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_5Reshape,locally_connected1d/strided_slice_5:output:0,locally_connected1d/Reshape_5/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"    <       А
+locally_connected1d/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    F       А
+locally_connected1d/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_6StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_6/stack:output:04locally_connected1d/strided_slice_6/stack_1:output:04locally_connected1d/strided_slice_6/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_6Reshape,locally_connected1d/strided_slice_6:output:0,locally_connected1d/Reshape_6/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*!
valueB"    F       А
+locally_connected1d/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    P       А
+locally_connected1d/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_7StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_7/stack:output:04locally_connected1d/strided_slice_7/stack_1:output:04locally_connected1d/strided_slice_7/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_7Reshape,locally_connected1d/strided_slice_7:output:0,locally_connected1d/Reshape_7/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*!
valueB"    P       А
+locally_connected1d/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    Z       А
+locally_connected1d/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_8StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_8/stack:output:04locally_connected1d/strided_slice_8/stack_1:output:04locally_connected1d/strided_slice_8/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_8Reshape,locally_connected1d/strided_slice_8:output:0,locally_connected1d/Reshape_8/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*!
valueB"    Z       А
+locally_connected1d/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    d       А
+locally_connected1d/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_9StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_9/stack:output:04locally_connected1d/strided_slice_9/stack_1:output:04locally_connected1d/strided_slice_9/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_9Reshape,locally_connected1d/strided_slice_9:output:0,locally_connected1d/Reshape_9/shape:output:0*
T0*+
_output_shapes
:         
a
locally_connected1d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Э
locally_connected1d/concatConcatV2$locally_connected1d/Reshape:output:0&locally_connected1d/Reshape_1:output:0&locally_connected1d/Reshape_2:output:0&locally_connected1d/Reshape_3:output:0&locally_connected1d/Reshape_4:output:0&locally_connected1d/Reshape_5:output:0&locally_connected1d/Reshape_6:output:0&locally_connected1d/Reshape_7:output:0&locally_connected1d/Reshape_8:output:0&locally_connected1d/Reshape_9:output:0(locally_connected1d/concat/axis:output:0*
N
*
T0*+
_output_shapes
:
         
а
)locally_connected1d/MatMul/ReadVariableOpReadVariableOp2locally_connected1d_matmul_readvariableop_resource*"
_output_shapes
:

*
dtype0╣
locally_connected1d/MatMulBatchMatMulV2#locally_connected1d/concat:output:01locally_connected1d/MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:
         l
locally_connected1d/ShapeShape#locally_connected1d/MatMul:output:0*
T0*
_output_shapes
:y
$locally_connected1d/Reshape_10/shapeConst*
_output_shapes
:*
dtype0*!
valueB"
          │
locally_connected1d/Reshape_10Reshape#locally_connected1d/MatMul:output:0-locally_connected1d/Reshape_10/shape:output:0*
T0*+
_output_shapes
:
         w
"locally_connected1d/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╢
locally_connected1d/transpose	Transpose'locally_connected1d/Reshape_10:output:0+locally_connected1d/transpose/perm:output:0*
T0*+
_output_shapes
:         
Ж
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       f
dense/Tensordot/ShapeShape!locally_connected1d/transpose:y:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╙
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ж
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ┤
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Л
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:а
dense/Tensordot/transpose	Transpose!locally_connected1d/transpose:y:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:         
Ь
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  Ь
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Х
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
о
2multi_level__block_attention/MatMul/ReadVariableOpReadVariableOp;multi_level__block_attention_matmul_readvariableop_resource*
_output_shapes

:*
dtype0╛
#multi_level__block_attention/MatMulBatchMatMulV2dense/BiasAdd:output:0:multi_level__block_attention/MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
▓
4multi_level__block_attention/MatMul_1/ReadVariableOpReadVariableOp=multi_level__block_attention_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0┬
%multi_level__block_attention/MatMul_1BatchMatMulV2dense/BiasAdd:output:0<multi_level__block_attention/MatMul_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
▓
4multi_level__block_attention/MatMul_2/ReadVariableOpReadVariableOp=multi_level__block_attention_matmul_2_readvariableop_resource*
_output_shapes

:*
dtype0┬
%multi_level__block_attention/MatMul_2BatchMatMulV2dense/BiasAdd:output:0<multi_level__block_attention/MatMul_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
и
/multi_level__block_attention/add/ReadVariableOpReadVariableOp8multi_level__block_attention_add_readvariableop_resource*
_output_shapes

:
*
dtype0╞
 multi_level__block_attention/addAddV2,multi_level__block_attention/MatMul:output:07multi_level__block_attention/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_1/ReadVariableOpReadVariableOp:multi_level__block_attention_add_1_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_1AddV2.multi_level__block_attention/MatMul_1:output:09multi_level__block_attention/add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_2/ReadVariableOpReadVariableOp:multi_level__block_attention_add_2_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_2AddV2.multi_level__block_attention/MatMul_2:output:09multi_level__block_attention/add_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
╟
%multi_level__block_attention/MatMul_3BatchMatMulV2$multi_level__block_attention/add:z:0&multi_level__block_attention/add_1:z:0*
T0*+
_output_shapes
:         

*
adj_y(e
#multi_level__block_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
З
!multi_level__block_attention/CastCast,multi_level__block_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: q
!multi_level__block_attention/SqrtSqrt%multi_level__block_attention/Cast:y:0*
T0*
_output_shapes
: ╝
$multi_level__block_attention/truedivRealDiv.multi_level__block_attention/MatMul_3:output:0%multi_level__block_attention/Sqrt:y:0*
T0*+
_output_shapes
:         

П
$multi_level__block_attention/SoftmaxSoftmax(multi_level__block_attention/truediv:z:0*
T0*+
_output_shapes
:         

и
/multi_level__block_attention/Mul/ReadVariableOpReadVariableOp8multi_level__block_attention_mul_readvariableop_resource*
_output_shapes

:

*
dtype0╞
 multi_level__block_attention/MulMul.multi_level__block_attention/Softmax:softmax:07multi_level__block_attention/Mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         

║
%multi_level__block_attention/MatMul_4BatchMatMulV2$multi_level__block_attention/Mul:z:0&multi_level__block_attention/add_2:z:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_3/ReadVariableOpReadVariableOp:multi_level__block_attention_add_3_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_3AddV2.multi_level__block_attention/MatMul_4:output:09multi_level__block_attention/add_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        п
conv1d/Conv1D/ExpandDims
ExpandDims&multi_level__block_attention/add_3:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
а
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╡
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:┴
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
О
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        s
activation/SigmoidSigmoidconv1d/Conv1D/Squeeze:output:0*
T0*+
_output_shapes
:         
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   |
flatten/ReshapeReshapeactivation/Sigmoid:y:0flatten/Const:output:0*
T0*'
_output_shapes
:         
q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :▒
global_average_pooling1d/MeanMeanconv1d/Conv1D/Squeeze:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         `
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Р
flatten_1/ReshapeReshape&global_average_pooling1d/Mean:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:         Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Л
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x
add/addAddV2flatten_1/Reshape:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentityadd/add:z:0^NoOp*
T0*'
_output_shapes
:         ┬
NoOpNoOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*^locally_connected1d/MatMul/ReadVariableOp3^multi_level__block_attention/MatMul/ReadVariableOp5^multi_level__block_attention/MatMul_1/ReadVariableOp5^multi_level__block_attention/MatMul_2/ReadVariableOp0^multi_level__block_attention/Mul/ReadVariableOp0^multi_level__block_attention/add/ReadVariableOp2^multi_level__block_attention/add_1/ReadVariableOp2^multi_level__block_attention/add_2/ReadVariableOp2^multi_level__block_attention/add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         d: : : : : : : : : : : : : : 2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2V
)locally_connected1d/MatMul/ReadVariableOp)locally_connected1d/MatMul/ReadVariableOp2h
2multi_level__block_attention/MatMul/ReadVariableOp2multi_level__block_attention/MatMul/ReadVariableOp2l
4multi_level__block_attention/MatMul_1/ReadVariableOp4multi_level__block_attention/MatMul_1/ReadVariableOp2l
4multi_level__block_attention/MatMul_2/ReadVariableOp4multi_level__block_attention/MatMul_2/ReadVariableOp2b
/multi_level__block_attention/Mul/ReadVariableOp/multi_level__block_attention/Mul/ReadVariableOp2b
/multi_level__block_attention/add/ReadVariableOp/multi_level__block_attention/add/ReadVariableOp2f
1multi_level__block_attention/add_1/ReadVariableOp1multi_level__block_attention/add_1/ReadVariableOp2f
1multi_level__block_attention/add_2/ReadVariableOp1multi_level__block_attention/add_2/ReadVariableOp2f
1multi_level__block_attention/add_3/ReadVariableOp1multi_level__block_attention/add_3/ReadVariableOp:S O
+
_output_shapes
:         d
 
_user_specified_nameinputs
Ъ
F
*__inference_flatten_1_layer_call_fn_318494

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╦
о
'__inference_conv1d_layer_call_fn_318442

inputsA
+conv1d_expanddims_1_readvariableop_resource:
identityИв"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : а
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:м
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        j
IdentityIdentityConv1D/Squeeze:output:0^NoOp*
T0*+
_output_shapes
:         
k
NoOpNoOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         
: 2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
о╬
─
A__inference_model_layer_call_and_return_conditional_losses_317983

inputsH
2locally_connected1d_matmul_readvariableop_resource:

9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:M
;multi_level__block_attention_matmul_readvariableop_resource:O
=multi_level__block_attention_matmul_1_readvariableop_resource:O
=multi_level__block_attention_matmul_2_readvariableop_resource:J
8multi_level__block_attention_add_readvariableop_resource:
L
:multi_level__block_attention_add_1_readvariableop_resource:
L
:multi_level__block_attention_add_2_readvariableop_resource:
J
8multi_level__block_attention_mul_readvariableop_resource:

L
:multi_level__block_attention_add_3_readvariableop_resource:
H
2conv1d_conv1d_expanddims_1_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:
5
'dense_1_biasadd_readvariableop_resource:
identityИв)conv1d/Conv1D/ExpandDims_1/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/Tensordot/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв)locally_connected1d/MatMul/ReadVariableOpв2multi_level__block_attention/MatMul/ReadVariableOpв4multi_level__block_attention/MatMul_1/ReadVariableOpв4multi_level__block_attention/MatMul_2/ReadVariableOpв/multi_level__block_attention/Mul/ReadVariableOpв/multi_level__block_attention/add/ReadVariableOpв1multi_level__block_attention/add_1/ReadVariableOpв1multi_level__block_attention/add_2/ReadVariableOpв1multi_level__block_attention/add_3/ReadVariableOpД
zero_padding1d/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       }
zero_padding1d/PadPadinputs$zero_padding1d/Pad/paddings:output:0*
T0*+
_output_shapes
:         e|
'locally_connected1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            ~
)locally_connected1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    
       ~
)locally_connected1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ═
!locally_connected1d/strided_sliceStridedSlicezero_padding1d/Pad:output:00locally_connected1d/strided_slice/stack:output:02locally_connected1d/strided_slice/stack_1:output:02locally_connected1d/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskv
!locally_connected1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ┤
locally_connected1d/ReshapeReshape*locally_connected1d/strided_slice:output:0*locally_connected1d/Reshape/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"    
       А
+locally_connected1d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_1StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_1/stack:output:04locally_connected1d/strided_slice_1/stack_1:output:04locally_connected1d/strided_slice_1/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_1Reshape,locally_connected1d/strided_slice_1:output:0,locally_connected1d/Reshape_1/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_2StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_2/stack:output:04locally_connected1d/strided_slice_2/stack_1:output:04locally_connected1d/strided_slice_2/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_2Reshape,locally_connected1d/strided_slice_2:output:0,locally_connected1d/Reshape_2/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"           А
+locally_connected1d/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    (       А
+locally_connected1d/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_3StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_3/stack:output:04locally_connected1d/strided_slice_3/stack_1:output:04locally_connected1d/strided_slice_3/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_3Reshape,locally_connected1d/strided_slice_3:output:0,locally_connected1d/Reshape_3/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"    (       А
+locally_connected1d/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    2       А
+locally_connected1d/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_4StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_4/stack:output:04locally_connected1d/strided_slice_4/stack_1:output:04locally_connected1d/strided_slice_4/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_4Reshape,locally_connected1d/strided_slice_4:output:0,locally_connected1d/Reshape_4/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*!
valueB"    2       А
+locally_connected1d/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    <       А
+locally_connected1d/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_5StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_5/stack:output:04locally_connected1d/strided_slice_5/stack_1:output:04locally_connected1d/strided_slice_5/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_5Reshape,locally_connected1d/strided_slice_5:output:0,locally_connected1d/Reshape_5/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"    <       А
+locally_connected1d/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    F       А
+locally_connected1d/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_6StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_6/stack:output:04locally_connected1d/strided_slice_6/stack_1:output:04locally_connected1d/strided_slice_6/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_6Reshape,locally_connected1d/strided_slice_6:output:0,locally_connected1d/Reshape_6/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*!
valueB"    F       А
+locally_connected1d/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    P       А
+locally_connected1d/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_7StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_7/stack:output:04locally_connected1d/strided_slice_7/stack_1:output:04locally_connected1d/strided_slice_7/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_7Reshape,locally_connected1d/strided_slice_7:output:0,locally_connected1d/Reshape_7/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*!
valueB"    P       А
+locally_connected1d/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    Z       А
+locally_connected1d/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_8StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_8/stack:output:04locally_connected1d/strided_slice_8/stack_1:output:04locally_connected1d/strided_slice_8/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_8Reshape,locally_connected1d/strided_slice_8:output:0,locally_connected1d/Reshape_8/shape:output:0*
T0*+
_output_shapes
:         
~
)locally_connected1d/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*!
valueB"    Z       А
+locally_connected1d/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"    d       А
+locally_connected1d/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ╒
#locally_connected1d/strided_slice_9StridedSlicezero_padding1d/Pad:output:02locally_connected1d/strided_slice_9/stack:output:04locally_connected1d/strided_slice_9/stack_1:output:04locally_connected1d/strided_slice_9/stack_2:output:0*
Index0*
T0*+
_output_shapes
:         
*

begin_mask*
end_maskx
#locally_connected1d/Reshape_9/shapeConst*
_output_shapes
:*
dtype0*!
valueB"       
   ║
locally_connected1d/Reshape_9Reshape,locally_connected1d/strided_slice_9:output:0,locally_connected1d/Reshape_9/shape:output:0*
T0*+
_output_shapes
:         
a
locally_connected1d/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Э
locally_connected1d/concatConcatV2$locally_connected1d/Reshape:output:0&locally_connected1d/Reshape_1:output:0&locally_connected1d/Reshape_2:output:0&locally_connected1d/Reshape_3:output:0&locally_connected1d/Reshape_4:output:0&locally_connected1d/Reshape_5:output:0&locally_connected1d/Reshape_6:output:0&locally_connected1d/Reshape_7:output:0&locally_connected1d/Reshape_8:output:0&locally_connected1d/Reshape_9:output:0(locally_connected1d/concat/axis:output:0*
N
*
T0*+
_output_shapes
:
         
а
)locally_connected1d/MatMul/ReadVariableOpReadVariableOp2locally_connected1d_matmul_readvariableop_resource*"
_output_shapes
:

*
dtype0╣
locally_connected1d/MatMulBatchMatMulV2#locally_connected1d/concat:output:01locally_connected1d/MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:
         l
locally_connected1d/ShapeShape#locally_connected1d/MatMul:output:0*
T0*
_output_shapes
:y
$locally_connected1d/Reshape_10/shapeConst*
_output_shapes
:*
dtype0*!
valueB"
          │
locally_connected1d/Reshape_10Reshape#locally_connected1d/MatMul:output:0-locally_connected1d/Reshape_10/shape:output:0*
T0*+
_output_shapes
:
         w
"locally_connected1d/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╢
locally_connected1d/transpose	Transpose'locally_connected1d/Reshape_10:output:0+locally_connected1d/transpose/perm:output:0*
T0*+
_output_shapes
:         
Ж
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       f
dense/Tensordot/ShapeShape!locally_connected1d/transpose:y:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╙
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ж
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ┤
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Л
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:а
dense/Tensordot/transpose	Transpose!locally_connected1d/transpose:y:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:         
Ь
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  Ь
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Х
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
о
2multi_level__block_attention/MatMul/ReadVariableOpReadVariableOp;multi_level__block_attention_matmul_readvariableop_resource*
_output_shapes

:*
dtype0╛
#multi_level__block_attention/MatMulBatchMatMulV2dense/BiasAdd:output:0:multi_level__block_attention/MatMul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
▓
4multi_level__block_attention/MatMul_1/ReadVariableOpReadVariableOp=multi_level__block_attention_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0┬
%multi_level__block_attention/MatMul_1BatchMatMulV2dense/BiasAdd:output:0<multi_level__block_attention/MatMul_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
▓
4multi_level__block_attention/MatMul_2/ReadVariableOpReadVariableOp=multi_level__block_attention_matmul_2_readvariableop_resource*
_output_shapes

:*
dtype0┬
%multi_level__block_attention/MatMul_2BatchMatMulV2dense/BiasAdd:output:0<multi_level__block_attention/MatMul_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
и
/multi_level__block_attention/add/ReadVariableOpReadVariableOp8multi_level__block_attention_add_readvariableop_resource*
_output_shapes

:
*
dtype0╞
 multi_level__block_attention/addAddV2,multi_level__block_attention/MatMul:output:07multi_level__block_attention/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_1/ReadVariableOpReadVariableOp:multi_level__block_attention_add_1_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_1AddV2.multi_level__block_attention/MatMul_1:output:09multi_level__block_attention/add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_2/ReadVariableOpReadVariableOp:multi_level__block_attention_add_2_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_2AddV2.multi_level__block_attention/MatMul_2:output:09multi_level__block_attention/add_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
╟
%multi_level__block_attention/MatMul_3BatchMatMulV2$multi_level__block_attention/add:z:0&multi_level__block_attention/add_1:z:0*
T0*+
_output_shapes
:         

*
adj_y(e
#multi_level__block_attention/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
З
!multi_level__block_attention/CastCast,multi_level__block_attention/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: q
!multi_level__block_attention/SqrtSqrt%multi_level__block_attention/Cast:y:0*
T0*
_output_shapes
: ╝
$multi_level__block_attention/truedivRealDiv.multi_level__block_attention/MatMul_3:output:0%multi_level__block_attention/Sqrt:y:0*
T0*+
_output_shapes
:         

П
$multi_level__block_attention/SoftmaxSoftmax(multi_level__block_attention/truediv:z:0*
T0*+
_output_shapes
:         

и
/multi_level__block_attention/Mul/ReadVariableOpReadVariableOp8multi_level__block_attention_mul_readvariableop_resource*
_output_shapes

:

*
dtype0╞
 multi_level__block_attention/MulMul.multi_level__block_attention/Softmax:softmax:07multi_level__block_attention/Mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         

║
%multi_level__block_attention/MatMul_4BatchMatMulV2$multi_level__block_attention/Mul:z:0&multi_level__block_attention/add_2:z:0*
T0*+
_output_shapes
:         
м
1multi_level__block_attention/add_3/ReadVariableOpReadVariableOp:multi_level__block_attention_add_3_readvariableop_resource*
_output_shapes

:
*
dtype0╠
"multi_level__block_attention/add_3AddV2.multi_level__block_attention/MatMul_4:output:09multi_level__block_attention/add_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        п
conv1d/Conv1D/ExpandDims
ExpandDims&multi_level__block_attention/add_3:z:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         
а
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ╡
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:┴
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:         
*
paddingSAME*
strides
О
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

¤        s
activation/SigmoidSigmoidconv1d/Conv1D/Squeeze:output:0*
T0*+
_output_shapes
:         
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    
   |
flatten/ReshapeReshapeactivation/Sigmoid:y:0flatten/Const:output:0*
T0*'
_output_shapes
:         
q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :▒
global_average_pooling1d/MeanMeanconv1d/Conv1D/Squeeze:output:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         `
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Р
flatten_1/ReshapeReshape&global_average_pooling1d/Mean:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:         Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Л
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x
add/addAddV2flatten_1/Reshape:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentityadd/add:z:0^NoOp*
T0*'
_output_shapes
:         ┬
NoOpNoOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*^locally_connected1d/MatMul/ReadVariableOp3^multi_level__block_attention/MatMul/ReadVariableOp5^multi_level__block_attention/MatMul_1/ReadVariableOp5^multi_level__block_attention/MatMul_2/ReadVariableOp0^multi_level__block_attention/Mul/ReadVariableOp0^multi_level__block_attention/add/ReadVariableOp2^multi_level__block_attention/add_1/ReadVariableOp2^multi_level__block_attention/add_2/ReadVariableOp2^multi_level__block_attention/add_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         d: : : : : : : : : : : : : : 2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2V
)locally_connected1d/MatMul/ReadVariableOp)locally_connected1d/MatMul/ReadVariableOp2h
2multi_level__block_attention/MatMul/ReadVariableOp2multi_level__block_attention/MatMul/ReadVariableOp2l
4multi_level__block_attention/MatMul_1/ReadVariableOp4multi_level__block_attention/MatMul_1/ReadVariableOp2l
4multi_level__block_attention/MatMul_2/ReadVariableOp4multi_level__block_attention/MatMul_2/ReadVariableOp2b
/multi_level__block_attention/Mul/ReadVariableOp/multi_level__block_attention/Mul/ReadVariableOp2b
/multi_level__block_attention/add/ReadVariableOp/multi_level__block_attention/add/ReadVariableOp2f
1multi_level__block_attention/add_1/ReadVariableOp1multi_level__block_attention/add_1/ReadVariableOp2f
1multi_level__block_attention/add_2/ReadVariableOp1multi_level__block_attention/add_2/ReadVariableOp2f
1multi_level__block_attention/add_3/ReadVariableOp1multi_level__block_attention/add_3/ReadVariableOp:S O
+
_output_shapes
:         d
 
_user_specified_nameinputs
С
K
/__inference_zero_padding1d_layer_call_fn_318144

inputs
identityu
Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       q
PadPadinputsPad/paddings:output:0*
T0*=
_output_shapes+
):'                           j
IdentityIdentityPad:output:0*
T0*=
_output_shapes+
):'                           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
╚
°
A__inference_dense_layer_call_and_return_conditional_losses_318358

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:         
К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         
c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         
z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         

 
_user_specified_nameinputs"┐-
saver_filename:0
Identity:0Identity_388"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╢
serving_defaultв
K
input_layer_1:
serving_default_input_layer_1:0         d7
add0
StatefulPartitionedCall:0         tensorflow/serving/predict:╟В
Г
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
е
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
▒
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel"
_tf_keras_layer
╗
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias"
_tf_keras_layer
┬
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
"1Annotation_embedding_q_weights
	1Wq_ld
"2Annotation_embedding_k_weights
	2Wk_ld
"3Annotation_embedding_v_weights
	3Wv_ld
4Epigenome_embedding_weights
4
Wepigenome
5
query_bias
5bq
6key_bias
6bk
7
value_bias
7bv
8output_bias
8bo"
_tf_keras_layer
╙
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
 @_jit_compiled_convolution_op"
_tf_keras_layer
е
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
е
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
е
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
е
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses

_kernel
`bias"
_tf_keras_layer
е
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
Ж
"0
)1
*2
13
24
35
46
57
68
79
810
?11
_12
`13"
trackable_list_wrapper
Ж
"0
)1
*2
13
24
35
46
57
68
79
810
?11
_12
`13"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╬
ltrace_0
mtrace_1
ntrace_2
otrace_32у
&__inference_model_layer_call_fn_316003
&__inference_model_layer_call_fn_317673
&__inference_model_layer_call_fn_317828
&__inference_model_layer_call_fn_317167└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 zltrace_0zmtrace_1zntrace_2zotrace_3
║
ptrace_0
qtrace_1
rtrace_2
strace_32╧
A__inference_model_layer_call_and_return_conditional_losses_317983
A__inference_model_layer_call_and_return_conditional_losses_318138
A__inference_model_layer_call_and_return_conditional_losses_317322
A__inference_model_layer_call_and_return_conditional_losses_317477└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 zptrace_0zqtrace_1zrtrace_2zstrace_3
╥B╧
!__inference__wrapped_model_315819input_layer_1"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·
	tdecay
ulearning_rate
vmomentum
wrho
xiter
"rms╥
)rms╙
*rms╘
1rms╒
2rms╓
3rms╫
4rms╪
5rms┘
6rms┌
7rms█
8rms▄
?rms▌
_rms▐
`rms▀"
	optimizer
,
yserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
є
trace_02╓
/__inference_zero_padding1d_layer_call_fn_318144в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ztrace_0
Р
Аtrace_02ё
J__inference_zero_padding1d_layer_call_and_return_conditional_losses_318150в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zАtrace_0
'
"0"
trackable_list_wrapper
'
"0"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
·
Жtrace_02█
4__inference_locally_connected1d_layer_call_fn_318224в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЖtrace_0
Х
Зtrace_02Ў
O__inference_locally_connected1d_layer_call_and_return_conditional_losses_318298в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЗtrace_0
0:.

2locally_connected1d/kernel
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
ь
Нtrace_02═
&__inference_dense_layer_call_fn_318328в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zНtrace_0
З
Оtrace_02ш
A__inference_dense_layer_call_and_return_conditional_losses_318358в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zОtrace_0
:2dense/kernel
:2
dense/bias
X
10
21
32
43
54
65
76
87"
trackable_list_wrapper
X
10
21
32
43
54
65
76
87"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
■
Фtrace_02▀
=__inference_multi_level__block_attention_layer_call_fn_318394Э
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zФtrace_0
Щ
Хtrace_02·
X__inference_multi_level__block_attention_layer_call_and_return_conditional_losses_318430Э
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zХtrace_0
M:K2;multi_level__block_attention/Annotation_embedding_q_weights
M:K2;multi_level__block_attention/Annotation_embedding_k_weights
M:K2;multi_level__block_attention/Annotation_embedding_v_weights
J:H

28multi_level__block_attention/Epigenome_embedding_weights
9:7
2'multi_level__block_attention/query_bias
7:5
2%multi_level__block_attention/key_bias
9:7
2'multi_level__block_attention/value_bias
::8
2(multi_level__block_attention/output_bias
'
?0"
trackable_list_wrapper
'
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
э
Ыtrace_02╬
'__inference_conv1d_layer_call_fn_318442в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЫtrace_0
И
Ьtrace_02щ
B__inference_conv1d_layer_call_and_return_conditional_losses_318454в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЬtrace_0
#:!2conv1d/kernel
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
ё
вtrace_02╥
+__inference_activation_layer_call_fn_318459в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zвtrace_0
М
гtrace_02э
F__inference_activation_layer_call_and_return_conditional_losses_318464в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zгtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
М
йtrace_02э
9__inference_global_average_pooling1d_layer_call_fn_318470п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zйtrace_0
з
кtrace_02И
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_318476п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zкtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
ю
░trace_02╧
(__inference_flatten_layer_call_fn_318482в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z░trace_0
Й
▒trace_02ъ
C__inference_flatten_layer_call_and_return_conditional_losses_318488в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▒trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▓non_trainable_variables
│layers
┤metrics
 ╡layer_regularization_losses
╢layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
Ё
╖trace_02╤
*__inference_flatten_1_layer_call_fn_318494в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╖trace_0
Л
╕trace_02ь
E__inference_flatten_1_layer_call_and_return_conditional_losses_318500в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╕trace_0
.
_0
`1"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╣non_trainable_variables
║layers
╗metrics
 ╝layer_regularization_losses
╜layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
ю
╛trace_02╧
(__inference_dense_1_layer_call_fn_318510в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╛trace_0
Й
┐trace_02ъ
C__inference_dense_1_layer_call_and_return_conditional_losses_318520в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┐trace_0
 :
2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
└non_trainable_variables
┴layers
┬metrics
 ├layer_regularization_losses
─layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
ъ
┼trace_02╦
$__inference_add_layer_call_fn_318526в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┼trace_0
Е
╞trace_02ц
?__inference_add_layer_call_and_return_conditional_losses_318532в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╞trace_0
 "
trackable_list_wrapper
v
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
11"
trackable_list_wrapper
0
╟0
╚1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 B№
&__inference_model_layer_call_fn_316003input_layer_1"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
°Bї
&__inference_model_layer_call_fn_317673inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
°Bї
&__inference_model_layer_call_fn_317828inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 B№
&__inference_model_layer_call_fn_317167input_layer_1"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
УBР
A__inference_model_layer_call_and_return_conditional_losses_317983inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
УBР
A__inference_model_layer_call_and_return_conditional_losses_318138inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ЪBЧ
A__inference_model_layer_call_and_return_conditional_losses_317322input_layer_1"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ЪBЧ
A__inference_model_layer_call_and_return_conditional_losses_317477input_layer_1"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
: (2decay
: (2learning_rate
: (2momentum
: (2rho
:	 (2RMSprop/iter
╤B╬
$__inference_signature_wrapper_317518input_layer_1"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
уBр
/__inference_zero_padding1d_layer_call_fn_318144inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■B√
J__inference_zero_padding1d_layer_call_and_return_conditional_losses_318150inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
шBх
4__inference_locally_connected1d_layer_call_fn_318224inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ГBА
O__inference_locally_connected1d_layer_call_and_return_conditional_losses_318298inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
┌B╫
&__inference_dense_layer_call_fn_318328inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_dense_layer_call_and_return_conditional_losses_318358inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
щBц
=__inference_multi_level__block_attention_layer_call_fn_318394x/0"Э
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
X__inference_multi_level__block_attention_layer_call_and_return_conditional_losses_318430x/0"Э
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
█B╪
'__inference_conv1d_layer_call_fn_318442inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_conv1d_layer_call_and_return_conditional_losses_318454inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▀B▄
+__inference_activation_layer_call_fn_318459inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_activation_layer_call_and_return_conditional_losses_318464inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
·Bў
9__inference_global_average_pooling1d_layer_call_fn_318470inputs"п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ХBТ
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_318476inputs"п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▄B┘
(__inference_flatten_layer_call_fn_318482inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
C__inference_flatten_layer_call_and_return_conditional_losses_318488inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▐B█
*__inference_flatten_1_layer_call_fn_318494inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_flatten_1_layer_call_and_return_conditional_losses_318500inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▄B┘
(__inference_dense_1_layer_call_fn_318510inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
C__inference_dense_1_layer_call_and_return_conditional_losses_318520inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
фBс
$__inference_add_layer_call_fn_318526inputs/0inputs/1"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 B№
?__inference_add_layer_call_and_return_conditional_losses_318532inputs/0inputs/1"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
R
╔	variables
╩	keras_api

╦total

╠count"
_tf_keras_metric
c
═	variables
╬	keras_api

╧total

╨count
╤
_fn_kwargs"
_tf_keras_metric
0
╦0
╠1"
trackable_list_wrapper
.
╔	variables"
_generic_user_object
:  (2total
:  (2count
0
╧0
╨1"
trackable_list_wrapper
.
═	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
::8

2&RMSprop/locally_connected1d/kernel/rms
(:&2RMSprop/dense/kernel/rms
": 2RMSprop/dense/bias/rms
W:U2GRMSprop/multi_level__block_attention/Annotation_embedding_q_weights/rms
W:U2GRMSprop/multi_level__block_attention/Annotation_embedding_k_weights/rms
W:U2GRMSprop/multi_level__block_attention/Annotation_embedding_v_weights/rms
T:R

2DRMSprop/multi_level__block_attention/Epigenome_embedding_weights/rms
C:A
23RMSprop/multi_level__block_attention/query_bias/rms
A:?
21RMSprop/multi_level__block_attention/key_bias/rms
C:A
23RMSprop/multi_level__block_attention/value_bias/rms
D:B
24RMSprop/multi_level__block_attention/output_bias/rms
-:+2RMSprop/conv1d/kernel/rms
*:(
2RMSprop/dense_1/kernel/rms
$:"2RMSprop/dense_1/bias/rmsЬ
!__inference__wrapped_model_315819w")*12356748?_`:в7
0в-
+К(
input_layer_1         d
к ")к&
$
addК
add         к
F__inference_activation_layer_call_and_return_conditional_losses_318464`3в0
)в&
$К!
inputs         

к ")в&
К
0         

Ъ В
+__inference_activation_layer_call_fn_318459S3в0
)в&
$К!
inputs         

к "К         
╟
?__inference_add_layer_call_and_return_conditional_losses_318532ГZвW
PвM
KЪH
"К
inputs/0         
"К
inputs/1         
к "%в"
К
0         
Ъ Ю
$__inference_add_layer_call_fn_318526vZвW
PвM
KЪH
"К
inputs/0         
"К
inputs/1         
к "К         й
B__inference_conv1d_layer_call_and_return_conditional_losses_318454c?3в0
)в&
$К!
inputs         

к ")в&
К
0         

Ъ Б
'__inference_conv1d_layer_call_fn_318442V?3в0
)в&
$К!
inputs         

к "К         
г
C__inference_dense_1_layer_call_and_return_conditional_losses_318520\_`/в,
%в"
 К
inputs         

к "%в"
К
0         
Ъ {
(__inference_dense_1_layer_call_fn_318510O_`/в,
%в"
 К
inputs         

к "К         й
A__inference_dense_layer_call_and_return_conditional_losses_318358d)*3в0
)в&
$К!
inputs         

к ")в&
К
0         

Ъ Б
&__inference_dense_layer_call_fn_318328W)*3в0
)в&
$К!
inputs         

к "К         
б
E__inference_flatten_1_layer_call_and_return_conditional_losses_318500X/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ y
*__inference_flatten_1_layer_call_fn_318494K/в,
%в"
 К
inputs         
к "К         г
C__inference_flatten_layer_call_and_return_conditional_losses_318488\3в0
)в&
$К!
inputs         

к "%в"
К
0         

Ъ {
(__inference_flatten_layer_call_fn_318482O3в0
)в&
$К!
inputs         

к "К         
╙
T__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_318476{IвF
?в<
6К3
inputs'                           

 
к ".в+
$К!
0                  
Ъ л
9__inference_global_average_pooling1d_layer_call_fn_318470nIвF
?в<
6К3
inputs'                           

 
к "!К                  ╢
O__inference_locally_connected1d_layer_call_and_return_conditional_losses_318298c"3в0
)в&
$К!
inputs         e
к ")в&
К
0         

Ъ О
4__inference_locally_connected1d_layer_call_fn_318224V"3в0
)в&
$К!
inputs         e
к "К         
└
A__inference_model_layer_call_and_return_conditional_losses_317322{")*12356748?_`Bв?
8в5
+К(
input_layer_1         d
p 

 
к "%в"
К
0         
Ъ └
A__inference_model_layer_call_and_return_conditional_losses_317477{")*12356748?_`Bв?
8в5
+К(
input_layer_1         d
p

 
к "%в"
К
0         
Ъ ╣
A__inference_model_layer_call_and_return_conditional_losses_317983t")*12356748?_`;в8
1в.
$К!
inputs         d
p 

 
к "%в"
К
0         
Ъ ╣
A__inference_model_layer_call_and_return_conditional_losses_318138t")*12356748?_`;в8
1в.
$К!
inputs         d
p

 
к "%в"
К
0         
Ъ Ш
&__inference_model_layer_call_fn_316003n")*12356748?_`Bв?
8в5
+К(
input_layer_1         d
p 

 
к "К         Ш
&__inference_model_layer_call_fn_317167n")*12356748?_`Bв?
8в5
+К(
input_layer_1         d
p

 
к "К         С
&__inference_model_layer_call_fn_317673g")*12356748?_`;в8
1в.
$К!
inputs         d
p 

 
к "К         С
&__inference_model_layer_call_fn_317828g")*12356748?_`;в8
1в.
$К!
inputs         d
p

 
к "К         є
X__inference_multi_level__block_attention_layer_call_and_return_conditional_losses_318430Ц123567485в2
+в(
&Ъ#
!К
x/0         

к "SвP
IвF
!К
0/0         

!К
0/1         


Ъ ╩
=__inference_multi_level__block_attention_layer_call_fn_318394И123567485в2
+в(
&Ъ#
!К
x/0         

к "EвB
К
0         

К
1         

▒
$__inference_signature_wrapper_317518И")*12356748?_`KвH
в 
Aк>
<
input_layer_1+К(
input_layer_1         d")к&
$
addК
add         ╙
J__inference_zero_padding1d_layer_call_and_return_conditional_losses_318150ДEвB
;в8
6К3
inputs'                           
к ";в8
1К.
0'                           
Ъ к
/__inference_zero_padding1d_layer_call_fn_318144wEвB
;в8
6К3
inputs'                           
к ".К+'                           