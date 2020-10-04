import sys
import cv2
if __name__ == '__main__':
    # If image path and f/q is not passed as command
    # line arguments, quit and display help message

    # speed-up using multithreads
    cv2.setUseOptimized(True);
    cv2.setNumThreads(4)  # 设置线程数

    # read image
    path = "./2.jpg"
    im = cv2.imread(path)

    # resize image 重置大小
    newHeight = 2000
    newWidth = int(im.shape[1]*200/im.shape[0])
    #     im = cv2.resize(im, (newWidth, newHeight))

    # create Selective Search Segmentation Object using default parameters
    # 使用默认参数创建一个选择性搜索的分割对象
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # set input image on which we will run segmentation
    ss.setBaseImage(im)


    model = "q"   # 快速模式 f 还是 高召回率模式 q
    # 三种模式，参考
    # https://docs.opencv.org/3.4/d6/d6d/classcv_1_1ximgproc_1_1segmentation_1_1SelectiveSearchSegmentation.html

    # Switch to fast but low recall Selective Search method
    # 快速模式，但是意味着低召回率
    if (model == 'f'):
        ss.switchToSelectiveSearchFast()

    # Switch to high recall but slow Selective Search method
    # 高召回率模式，但是速度较慢
    elif (model == 'q'):
        ss.switchToSelectiveSearchQuality()
    elif (model == "ToString"):
        ss.switchToSingleStrategy()
    # if argument is neither f nor q print help message
    else:
        print("plase set model!")

    # run selective search segmentation on input image
    # 运行选择性搜索算法，返回他们的可能边框
    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))

    # number of region proposals to show
    numShowRects = 100
    # increment to increase/decrease total number of reason proposals to be shown
    increment = 20

    while True:
        # create a copy of original image
        # 复制一份图片
        imOut = im.copy()

        # itereate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            # 绘制边框
            if (i < numShowRects):
                x, y, w, h = rect
                cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break

        # show output
        cv2.imshow("Output", imOut)
        cv2.imwrite("Output.jpg", imOut)

        # record key press
        k = cv2.waitKey(0) & 0xFF
        # m is pressed  m  键增加要显示的矩形框数
        if k == 109:
            # increase total number of rectangles to show by increment
            numShowRects += increment

        # l is pressed  l 键减少要显示的矩形框数
        elif k == 108 and numShowRects > increment:
            # decrease total number of rectangles to show by increment
            numShowRects -= increment

        # q is pressed  q 键退出
        elif k == 113:
            break

    # close image show window
    cv2.destroyAllWindows()