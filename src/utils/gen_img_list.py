# #!/usr/bin/env python

# '''
# MIT License

# Copyright (c) 2021 Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# '''

# from os.path import join
# import glob
# import argparse
# from natsort import natsorted

# def has_file_allowed_extension(filename, extensions):
#     filename_lower = filename.lower()
#     return any(filename_lower.endswith(ext) for ext in extensions)

# IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

# parser = argparse.ArgumentParser(description='genImageList')
# parser.add_argument('--dataset_root_dir', type=str,
#                     help='Path to folder where the images are saved.')
# parser.add_argument('--out_file', type=str,
#                     help='File to which the image list should be saved to.')

# if __name__ == "__main__":
#     opt = parser.parse_args()
#     image_list = []
#     for fname in natsorted(glob.glob(opt.dataset_root_dir + '/*.*')):
#         if has_file_allowed_extension(fname, IMG_EXTENSIONS):
#             path = join(opt.dataset_root_dir, fname)
#             image_list.append(path)

#     with open(opt.out_file, 'w') as out_file:
#         for image_name in image_list:
#             out_file.write(join(opt.dataset_root_dir, image_name) + '\n')

#     print('Done')
# import os  # 导入os模块
 

# total_img = []

# def search_file(start_dir):
#     global total_img
#     img_list = []
#     extend_name = ['.jpg', '.png', '.gif']  # 图片格式，可以添加其他图片格式
#     os.chdir(start_dir)  # 改变当前工作目录到指定的路径
 
#     for each_file in os.listdir(os.curdir):
#         # listdir()返回指定的文件夹包含的文件或文件夹的名字的列表 curdir表示当前工作目录
#         # print('each', each_file)
#         img_prop = os.path.splitext(each_file)
#         if img_prop[1] in extend_name:
#             img_list.append(os.getcwd() + os.sep + each_file + os.linesep)
#             # os.getcwd()获得当前路径 os.sep分隔符 os.linesep换行符
 
#         if os.path.isdir(each_file):  # isdir()判断是否为文件夹
#             search_file(each_file)  # 递归搜索子文件夹下的图片
#             os.chdir(os.pardir)  # 返回上一级工目录
#     total_img.append()
#     with open('./cmu.txt', 'w') as file_obj:  # 此处修改输出文本文件目录及名称
#         file_obj.writelines(img_list)  # writelines按行写入数据

# search_file('/root/halo/datasets/Extended-CMU-Seasons')


from pathlib import Path
root_dir = '/root/halo/datasets/Robotcar-Seasons/images/overcast-reference'
# ''
img = []
for folder in Path(root_dir).iterdir():
    print('floder', folder)
    a = list(map(str, list(folder.rglob("*.jpg"))))
    # b = list(map(str, list(folder.rglob("*.png"))))

    img.extend(a)
print('total imgs', len(img))

with open('./robotcar_v1_db.txt', 'w') as f:
    for i in img:
        f.write(i + '\n')