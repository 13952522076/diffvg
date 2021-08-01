import cv2
import os

dir = "/Users/melody/Downloads/diffvg_html/images_resized"
saved_dir = "/Users/melody/Downloads/diffvg_html/images_resized_2"
resize_factors = [1,2,3,4,5,6,7,8]


if not os.path.exists(saved_dir):
    os.mkdir(saved_dir)

for s in os.listdir(dir):
    if s.endswith("jpg") or s.endswith("jpeg") or s.endswith("png"):
        filename = s.split('.')[0]   # 63-1
        base_name = filename.split('-')[0] # 63
        src = cv2.imread(os.path.join(dir, s), cv2.IMREAD_UNCHANGED)
        base_src = cv2.imread(os.path.join(dir, base_name+'-1.png'), cv2.IMREAD_UNCHANGED)
        height, width = base_src.shape[0], base_src.shape[1]
        print(f"input image: {s} [width: {width}, height: {height}]")

        dsize = (width, height)
        output = cv2.resize(src, dsize)
        save_file_name = filename +  "-recovered.png"
        cv2.imwrite(os.path.join(saved_dir, save_file_name), output)

        # for r in resize_factors:
        #     dsize = (int(width//r), int(height//r))
        #     output = cv2.resize(src, dsize)
        #     save_file_name = filename+"-"+str(r)+".png"
        #     cv2.imwrite(os.path.join(saved_dir, save_file_name), output)
