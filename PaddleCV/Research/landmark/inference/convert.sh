#res152_softmax_v1
python convert_binary_model.py --model='ResNet152_vd_fc' --pretrained_model=pretrained_models/res152_softmax_v1/ --binary_model=./binary_models/res152_softmax_v1 --image_shape=3,224,224 --task_mode='classification'

#res152_softmax_v2
python convert_binary_model.py --model='ResNet152_vd' --pretrained_model=pretrained_models/res152_softmax_v2/ --binary_model=./binary_models/res152_softmax_v2 --image_shape=3,224,224 --task_mode='classification'

#res152_arcmargin
python convert_binary_model.py --model='ResNet152_vd_v0_embedding' --pretrained_model=pretrained_models/res152_arcmargin/ --binary_model=./binary_models/res152_arcmargin --image_shape=3,448,448 --task_mode='retrieval'

#res152_arcmargin_index
python convert_binary_model.py --model='ResNet152_vd_v0_embedding' --pretrained_model=pretrained_models/res152_arcmargin_index/ --binary_model=./binary_models/res152_arcmargin_index --image_shape=3,448,448 --task_mode='retrieval'

#res152_npairs
python convert_binary_model.py --model='ResNet152_vd_v0_embedding' --pretrained_model=pretrained_models/res152_npairs/ --binary_model=./binary_models/res152_npairs --image_shape=3,448,448 --task_mode='retrieval'

#res200_arcmargin
python convert_binary_model.py --model='ResNet200_vd_embedding' --pretrained_model=pretrained_models/res200_arcmargin/ --binary_model=./binary_models/res200_arcmargin --image_shape=3,448,448 --task_mode='retrieval'

#se_x152_arcmargin
python convert_binary_model.py --model='SE_ResNeXt152_64x4d_vd_embedding' --pretrained_model=pretrained_models/se_x152_arcmargin/ --binary_model=./binary_models/se_x152_arcmargin --image_shape=3,448,448 --task_mode='retrieval'

#inceptionv4_arcmargin
python convert_binary_model.py --model='InceptionV4_embedding' --pretrained_model=pretrained_models/inceptionv4_arcmargin --binary_model=./binary_models/inceptionv4_arcmargin --image_shape=3,555,555 --task_mode='retrieval'
