# Test a single image
python demo/restoration_single_face_demo.py --config configs/panini_mfr.py --checkpoint checkpoint/panini_mfr_latest.pth --img_path examples/MFR/00001.png --save_path examples/MFR_result/00001.png
python demo/restoration_single_face_demo.py --config configs/panini_sr.py --checkpoint checkpoint/panini_sr_latest.pth --img_path examples/SR/00001.png --save_path examples/SR_result/00001.png

# Test a directory with images
python demo/restoration_dir_face_demo.py --config configs/panini_mfr.py --checkpoint checkpoint/panini_mfr_latest.pth --img_path examples/MFR --save_path examples/MFR_result
python demo/restoration_dir_face_demo.py --config configs/panini_sr.py --checkpoint my_project/panini_sr/latest.pth --img_path examples/SR --save_path examples/SR_result