训练
python train_unet_h5.py --h5 "E:\mhw\1\pore\dataset_all_32.h5" --out_dir "E:\mhw\1\pore\runs\unet_all32" --epochs 200 --batch_size 64 --balanced_sampler --use_softplus --lambda_q 10 --lambda_neg 1.0

测试
python infer_unet_h5.py `
    --h5 "E:\mhw\1\pore\dataset_all_32.h5" `
    --ckpt "E:\mhw\1\pore\runs\unet_all\best.pt" `
    --out_dir "E:\mhw\1\pore\runs\unet_all\infer" `
    --save_field 1
归一化
python build_hdf5_dataset.py --root "E:\mhw\1\pore\data" --rocks 1 2 3 4 5 6 --out_h5 "E:\mhw\1\pore\dataset_all_32.h5" --raw_patch 32 --patch 32 --stride_raw 4 --R1 8 --buffer 1024