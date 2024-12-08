#!/usr/bin/env bash
DEVICE=0

echo ""
echo "-------------------------------------------------"
echo "| Train Xception on DFDC                         |"
echo "-------------------------------------------------"
# put your DFDC source directory path for the extracted faces and Dataframe and uncomment the following line
DFDC_FACES_DIR=D:\\DFDetect\\fakedetector-main\\faceout
DFDC_FACES_DF=D:\\DFDetect\\fakedetector-main\\faceout\\faces_df.pkl
python train_binclass.py \
--net Xception \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 30000 \
--seed 41 \
--attention \
--device $DEVICE




echo ""
echo "-------------------------------------------------"
echo "| Train EfficientNetB4 on DFDC                   |"
echo "-------------------------------------------------"
# put your DFDC source directory path for the extracted faces and Dataframe and uncomment the following line
DFDC_FACES_DIR=D:\\DFDetect\\fakedetector-main\\faceout
DFDC_FACES_DF=D:\\DFDetect\\fakedetector-main\\faceout\\faces_df.pkl
python train_binclass.py \
--net EfficientNetB4 \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 30000 \
--seed 41 \
--attention \
--device $DEVICE



--device $DEVICE


echo ""
echo "-------------------------------------------------"
echo "| Train EfficientNetAutoAttB4 on DFDC           |"
echo "-------------------------------------------------"
# put your DFDC source directory path for the extracted faces and Dataframe and uncomment the following line
DFDC_FACES_DIR=D:\\DFDetect\\fakedetector-main\\faceout
DFDC_FACES_DF=D:\\DFDetect\\fakedetector-main\\faceout\\faces_df.pkl
python train_binclass.py \
--net EfficientNetAutoAttB4 \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 30000 \
--seed 41 \
--attention \
--device $DEVICE


echo ""
echo "-------------------------------------------------"
echo "| Train EfficientNetAutoAttB4 on DFDC  (tuning) |"
echo "-------------------------------------------------"
# put your DFDC source directory path for the extracted faces and Dataframe and uncomment the following line
DFDC_FACES_DIR=D:\\DFDetect\\fakedetector-main\\faceout
DFDC_FACES_DF=D:\\DFDetect\\fakedetector-main\\faceout\\faces_df.pkl
python train_binclass.py \
--net EfficientNetAutoAttB4 \
--traindb dfdc-35-5-10 \
--valdb dfdc-35-5-10 \
--dfdc_faces_df_path $DFDC_FACES_DF \
--dfdc_faces_dir $DFDC_FACES_DIR \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 50 \
--patience 10 \
--maxiter 5000 \
--seed 41 \
--attention \
--init weights/binclass/net-EfficientNetB4_traindb-dfdc-35-5-10_face-scale_size-224_seed-41/bestval.pth \
--suffix finetuning \
--device $DEVICE
