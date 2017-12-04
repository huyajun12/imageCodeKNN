# # 江苏苏州
# img_array = numpy.array(im_obj)
# bin_list = []
# im_array = img_array[2:24, 1:69]
# for line in im_array:
#     sub_bin = []
#     for i in line:
#         if 200 < i[0] <= 255 and 200 < i[1] <= 255 and 200 < i[2] <=255:
#             sub_bin.append(0)
#
#         # elif 90 <= i[0] < 130 and 90 <= i[1] < 130 and 90 <= i[2] < 130:
#         #     value.append(0)
#
#         elif 0 <= i[0] < 20 and 0 <= i[1] < 20 and 0 <= i[2] < 20:
#             sub_bin.append(0)
#
#         else:
#             sub_bin.append(1)
#     bin_list.append(sub_bin)

# 广东东莞
# for line in img_array:
#     sub_bin = []
#     for i in line:
#         if 150 < i[0] <= 255 and 150 < i[1] <= 255 and 150 < i[2] <= 255:
#             sub_bin.append(0)
#
#         else:
#             sub_bin.append(1)
#         if 200 < i[0] <= 255 and 200 < i[1] <= 255 and 200 < i[2] <= 255:
#             sub_bin.append(0)
#
#         elif 90 < i[0] < 132 and 90 < i[1] < 132 and 90 < i[2] < 132:
#             sub_bin.append(0)
#     bin_list.append(sub_bin)

# phoneItem
# img_array = img_array[5:24, 3:(-2)]
# width = img_array.shape[1]  # 列
# height = img_array.shape[0]  # 行
# im_array = img_array[2:height-1, 1:width-1]
# for line in im_array:
#     sub_bin = []
#     for i in line:
#         if 150 < i[0] <= 255 and 150 < i[1] <= 255 and 150 < i[2] <= 255:
#             sub_bin.append(0)
#
#         else:
#             sub_bin.append(1)
#
#     bin_list.append(sub_bin)

# #51credit
# width = img_array.shape[1]  # 列
# height = img_array.shape[0]  # 行
# im_array = img_array[1:height-1, 1:width-1] #去边框
# for line in im_array:
#     sub_bin = []
#     for i in line:
#         if 140 < i[0] <= 255 and 140 < i[1] <= 255 and 140 < i[2] <= 255:
#             sub_bin.append(0)
#
#         else:
#             sub_bin.append(1)
#
#     bin_list.append(sub_bin)


# paiLaiDai
# im_obj = im_obj.filter(ImageFilter.MedianFilter())
# enhancer = ImageEnhance.Contrast(im_obj)
# img1 = enhancer.enhance(1.5)
# img2 = img1.convert('L')  # 转换成黑白
# img_array = numpy.array(img2)
# bin_list = []
# width = img_array.shape[1]  # 列
# height = img_array.shape[0]  # 行
#
# for line in img_array:
#     sub_bin = []
#     for i in line:
#         if 140 < i:
#             sub_bin.append(0)
#         else:
#             sub_bin.append(1)
#
#     bin_list.append(sub_bin)
#
# for j in range(height):
#     for i in range(width):
#         if 0 < i < width - 1 and 0 < j < height - 1 and bin_list[j][i] == 0 and bin_list[j][i - 1] == 1 and \
#                         bin_list[j][i + 1] == 1 and sum(
#             [bin_list[j - 1][i - 1], bin_list[j - 1][i], bin_list[j - 1][i + 1],
#              bin_list[j][i - 1], bin_list[j - 1][i + 1], bin_list[j + 1][i - 1], bin_list[j + 1][i],
#              bin_list[j + 1][i + 1]]) == 4:
#             bin_list[j][i] = 1
#             bin_list[j + 1][i] = 1




# CT_hainan
# im_array = numpy.array(im_obj)
# bin_list = []
# for line in im_array:
#     sub_bin = []
#     for i in line:
#         if (160 <= i[0] <= 255 or 16 <= i[0]) and 160 <= i[1] <= 255 and (160 <= i[2] <= 255 or 139 == i[2]):
#             sub_bin.append(0)
#
#         else:
#             sub_bin.append(1)
#     bin_list.append(sub_bin)

# CT_xinjaing
# img_array = numpy.array(im_obj)
# im_array = img_array[6:26, 14:84]
# bin_list = []
# for line in im_array:
#     sub_bin = []
#     for i in line:
#         if 210 <= i[0] and 210 <= i[1] and 110 <= i[2]:
#             sub_bin.append(0)
#         elif i[0] <= 30 and i[1] <= 20 and i[2] <= 20:
#             sub_bin.append(0)
#         else:
#             sub_bin.append(1)
#     bin_list.append(sub_bin)


#CT_jiangsu
# def process_img(im_obj):
#     '''
#     :param im_obj: 图片对象
#     :return: 二值化后的数组
#     '''
#     img_array = numpy.array(im_obj)
#     l0 = img_array.shape[0]
#     l1 = img_array.shape[1]
#     for i in range(l0):
#         for j in range(l1):
#             if 150 <= img_array[i][j][0] and 160 <= img_array[i][j][1] and 160 <= img_array[i][j][2]:
#                 img_array[i][j][0] = 255
#                 img_array[i][j][1] = 255
#                 img_array[i][j][2] = 255
#     img_obj2 = Image.fromarray(img_array)
#     im_obj3 = img_obj2.filter(ImageFilter.MedianFilter())
#     # im_obj3.save(r'G:\PycharmProjects\machineLearning\imageCodeKNN\CT_jiangsu\imagesArray\0.jpg')
#     #img_obj2.show()
#     img_array2 = numpy.array(im_obj3)
#
#     bin_list = []
#     for line in img_array2:
#         sub_bin = []
#         for i in line:
#             if 150 <= i[0] and 160 <= i[1] and 160 <= i[2]:
#                 sub_bin.append(0)
#             else:
#                 sub_bin.append(1)
#         bin_list.append(sub_bin)
#
#     return numpy.array(bin_list)



