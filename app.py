from flask import Flask, flash, render_template, redirect, url_for, request
from werkzeug.utils import secure_filename
from PIL import Image
from cv2 import cv2
from math import cos, sqrt, pi
import os
import numpy as np
import pandas as pd
import pywt
import pywt.data
import seaborn as sns

app = Flask(__name__)
app.secret_key = "image-processing"

UPLOAD_FOLDER = "static/uploads/"
RESULT_FOLDER = "static/results/"
STAT_FOLDER = "static/stats"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['STAT_FOLDER'] = STAT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 24 * 512 * 512

ALLOWED_EXTENSIONS = set(['bmp'])


def dct_algorithm(img_name, m, q_level):
    path = img_name
    img1 = cv2.imread(path)

    # Converting BGR channels to YCRCB channels
    ycrcb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
    y = ycrcb[:, :, 0]
    cr1 = ycrcb[:, :, 1]
    cb1 = ycrcb[:, :, 2]

    # Sub sampling cr & cb channels to half
    h, w = cr1.shape
    y_arr = np.float32(y)
    cr_arr = np.float32(cr1)
    cb_arr = np.float32(cb1)
    q_y = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56], [
                        14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77], [
                        24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.int32)
    q_c = np.array([[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
                    [24, 26, 56, 99, 99, 99, 99, 99], [
                        47, 66, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99], [
                        99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]], dtype=np.int32)

    # Compression function
    def DCT_main(img_name, img_arr, m, q_matrix, q_level):
        r, c = img_arr.shape
        img_arr = img_arr - 128
        dct2 = cv2.dct(img_arr)
        qnt_arr = np.zeros((r, c), dtype=np.int32)
        deqnt_arr = np.zeros((r, c), dtype=np.int32)

        def matrix(q_matrix, q_level):
            if q_level == 50:
                return q_matrix
            elif q_level < 50:
                q = 50/q_level
                for i in range(8):
                    for j in range(8):
                        q_matrix[i, j] = q_matrix[i, j]*q
                        if q_matrix[i, j] > 255:
                            q_matrix[i, j] = 255
            else:
                q = (100-q_level)/50
                for i in range(8):
                    for j in range(8):
                        q_matrix[i, j] = q_matrix[i, j]*q
            return q_matrix

        def quantization(dct_mask, q_matrix, qnt_mask, deqnt_mask):
            for i in range(8):
                for j in range(8):
                    qnt_mask[i, j] = dct_mask[i, j]/q_matrix[i, j]

            deqnt_mask = qnt_mask*q_matrix

            return qnt_mask, deqnt_mask

        x1 = 0
        y1 = 0
        q_mat = matrix(q_matrix, q_level)
        while x1 < r:
            x2 = x1 + m
            while y1 < c:
                y2 = y1 + m
                qnt, deqnt = quantization(dct2[x1:x2, y1:y2], q_mat, np.zeros(
                    (m, m), dtype=np.int32), np.zeros((m, m), dtype=np.int32))
                qnt_arr[x1:x2, y1:y2] = qnt
                deqnt_arr[x1:x2, y1:y2] = deqnt
                y1 += m
            y1 = 0
            x1 += m

        idct = cv2.idct(deqnt_arr.astype('float32'))
        idct = idct + 128.0
        idct = idct.astype('uint8')
        dct2 = dct2.astype('int32')

        return idct

    y_idct = DCT_main(img_name, y_arr, m, q_y, q_level)
    cr_idct = DCT_main(img_name, cr_arr, m, q_c, q_level)
    cb_idct = DCT_main(img_name, cb_arr, m, q_c, q_level)

    # Computing MSE and PSNR (image quality parameters)
    i = 0
    j = 0
    sq_error_y = 0
    sq_error_cr = 0
    sq_error_cb = 0
    while i < h:
        while j < w:
            sq_error_y += ((y[i][j] - y_idct[i][j])**2)/(h*w)
            sq_error_cr += ((cr1[i][j] - cr_idct[i][j])**2)/(h*w)
            sq_error_cb += ((cb1[i][j] - cb_idct[i][j])**2)/(h*w)
            j += 1
        j = 0
        i += 1

    MSE = sq_error_y+sq_error_cr+sq_error_cb

    PSNR = 10*np.log10(255*765/MSE)

    # Combining y,cr,cb channels
    ycrcbo = cv2.merge((y_idct, cr_idct, cb_idct))
    bgro = cv2.cvtColor(ycrcbo, cv2.COLOR_YCR_CB2BGR)

    file = request.files['file']
    name_image = file.filename
    result_image = name_image + '_' + str(q_level) + '%' + '_DCT_compress.png'
    if not cv2.imwrite(os.path.join(RESULT_FOLDER, result_image), bgro):
        raise Exception("Could not write image")

    original_size = os.path.getsize(UPLOAD_FOLDER + name_image)
    compressed_size = os.path.getsize(RESULT_FOLDER + result_image)
    CR = original_size/compressed_size

    compress = os.path.join(RESULT_FOLDER, result_image)

    MSE = round(MSE, 5)
    PSNR = round(PSNR, 5)
    CR = round(CR, 5)

    return MSE, PSNR, CR, compress


def dwt_algorithm(img_name, th):
    path = img_name
    img1 = cv2.imread(path)
    img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    h, w = img2.shape
    h = int(h)
    w = int(w)

    coeff = pywt.dwt2(img2, 'haar', mode='periodization')
    A, (H, V, D) = coeff

    def threshold(img_arr, th):
        r, c = img_arr.shape

        i = 0
        j = 0
        while i < r:
            while j < c:
                if abs(img_arr[i, j]) < th:
                    img_arr[i, j] = 0
                j += 1
            j = 0
            i += 1
        return img_arr

    H1 = threshold(H, th)
    V1 = threshold(V, th)
    D1 = threshold(D, th)

    th_coeff = A, (H1, V1, D1)
    idwt = pywt.idwt2(th_coeff, 'haar', mode='periodization')
    idwt = idwt.astype('uint8')

    file = request.files['file']
    name_image = file.filename
    result_image = name_image + '_' + str(th) + '_DWT_compress.png'
    if not cv2.imwrite(os.path.join(RESULT_FOLDER, result_image), idwt):
        raise Exception("Could not write image")

    x = 0
    y = 0
    MSE = 0
    while x < h:
        while y < w:
            MSE += ((img2[x][y] - idwt[x][y])**2)/(h*w)
            y += 1
        y = 0
        x += 1

    PSNR = 10*np.log10(255*255/MSE)

    # Calculating compression ratio
    original_size = os.path.getsize(UPLOAD_FOLDER + name_image)
    compressed_size = os.path.getsize(RESULT_FOLDER + result_image)
    CR = original_size/compressed_size

    compress = os.path.join(RESULT_FOLDER, result_image)

    MSE = round(MSE, 5)
    PSNR = round(PSNR, 5)
    CR = round(CR, 5)

    return MSE, PSNR, CR, compress


def btc_algorithm(img_name, m):

    path = img_name
    img = cv2.imread(path)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, c, d = img1.shape

    R = img1[:, :, 0]
    G = img1[:, :, 1]
    B = img1[:, :, 2]

    # Defining BTC compression function
    def color_compress(channel, m, r, c, op_arr):
        x = 0
        y = 0
        m2 = m**2
        while x < r:
            while y < c:
                mean = 0
                std = 0
                q = 0

                # Calculating mean(first moment)
                for i in range(x, x+m):
                    for j in range(y, y+m):
                        mean += channel[i][j]/m2

                # Calculating absolute moments
                for i in range(x, x+m):
                    for j in range(y, y+m):
                        if channel[i][j] >= mean:
                            q += 1
                            std += (channel[i][j]-mean)**2
                std = (std/m2)**0.5
                # When all pixel values in a window are same
                if q == m2:
                    y += m
                    continue

                # Calculating Low & High values for output image
                low = mean - ((q/(m2-q))**0.5)*std
                high = mean + (((m2-q)/q)**0.5)*std

                # Assigning the low and high values to bitmap(Low to 0 and High to 1)
                for i in range(x, x+m):
                    for j in range(y, y+m):
                        if channel[i][j] >= mean:
                            op_arr[i][j] = high
                        else:
                            op_arr[i][j] = low

                y += m
            y = 0
            x += m

        return op_arr

    RO = color_compress(R, m, r, c, np.zeros((r, c), dtype='uint8'))
    GO = color_compress(G, m, r, c, np.zeros((r, c), dtype='uint8'))
    BO = color_compress(B, m, r, c, np.zeros((r, c), dtype='uint8'))

    # Merging compressed RGB channels
    RGBO = cv2.merge((RO, GO, BO))
    RGBO2 = cv2.cvtColor(RGBO, cv2.COLOR_RGB2BGR)

    # Saving the images to a directory
    file = request.files['file']
    name_image = file.filename
    result_image = name_image + '_' + str(m) + '_BTC_compress.png'
    if not cv2.imwrite(os.path.join(RESULT_FOLDER, result_image), RGBO2):
        raise Exception("Could not write image")

    i = 0
    j = 0
    sq_error = 0
    while i < r:
        while j < c:
            sq_error += (R[i][j] - RO[i][j])**2
            sq_error += (G[i][j] - GO[i][j])**2
            sq_error += (B[i][j] - BO[i][j])**2
            j += 1
        j = 0
        i += 1

    MSE = sq_error/(3*r*c)

    PSNR = np.log10(255*765/abs(MSE))

    original_size = os.path.getsize(UPLOAD_FOLDER + name_image)
    compressed_size = os.path.getsize(RESULT_FOLDER + result_image)
    CR = original_size/compressed_size

    compress = os.path.join(RESULT_FOLDER, result_image)

    MSE = round(MSE, 5)
    PSNR = round(PSNR, 5)
    CR = round(CR, 5)

    return abs(MSE), PSNR, CR, compress


def ambtc_algorithm(img_name, mi, mq):

    path = img_name
    img = cv2.imread(path)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, c, d = img1.shape

    R = img1[:, :, 0]
    G = img1[:, :, 1]
    B = img1[:, :, 2]

    # Defining AMBTC compression function
    def color_compress(channel, m, r, c, op_arr):
        x = 0
        y = 0
        m2 = m**2
        while x < r:
            while y < c:
                mean = 0
                High = 0
                Low = 0
                q = 0

                # Calculating mean(first moment)
                for i in range(x, x+m):
                    for j in range(y, y+m):
                        mean += channel[i][j]/m2

                # Calculating absolute moments
                for i in range(x, x+m):
                    for j in range(y, y+m):
                        if channel[i][j] >= mean:
                            High += channel[i, j]
                            q += 1
                        else:
                            Low += channel[i, j]

                # When all pixel values in a window are same
                if q == m2:
                    y += m
                    continue
                h = High/q
                l = Low/((m**2)-q)

                # Assigning the low and high values to bitmap(Low to 0 and High to 1)
                for i in range(x, x+m):
                    for j in range(y, y+m):
                        if channel[i][j] >= mean:
                            op_arr[i][j] = h
                        else:
                            op_arr[i][j] = l

                y += m
            y = 0
            x += m

        return op_arr

    GO = color_compress(G, mi, r, c, np.zeros((r, c), dtype='uint8'))
    BO = color_compress(B, mq, r, c, np.zeros((r, c), dtype='uint8'))

    RGBO = cv2.merge((R, GO, BO))
    RGBO2 = cv2.cvtColor(RGBO, cv2.COLOR_RGB2BGR)

    # Saving the images to a directory
    file = request.files['file']
    name_image = file.filename
    result_image = name_image + '_' + str(mq) + '_AMBTC_compress.png'
    if not cv2.imwrite(os.path.join(RESULT_FOLDER, result_image), RGBO2):
        raise Exception("Could not write image")

    i = 0
    j = 0
    sq_error = 0
    while i < r:
        while j < c:
            sq_error += (R[i][j] - R[i][j])**2
            sq_error += (G[i][j] - GO[i][j])**2
            sq_error += (B[i][j] - BO[i][j])**2
            j += 1
        j = 0
        i += 1

    MSE = (sq_error/(3*r*c))

    PSNR = np.log10(255*765/abs(MSE))

    original_size = os.path.getsize(UPLOAD_FOLDER + name_image)
    compressed_size = os.path.getsize(RESULT_FOLDER + result_image)
    CR = original_size/compressed_size

    compress = os.path.join(RESULT_FOLDER, result_image)

    MSE = round(MSE, 5)
    PSNR = round(PSNR, 5)
    CR = round(CR, 5)

    return abs(MSE), PSNR, CR, compress


def svd_algorithm(img_name, sv):
    path = img_name
    img = cv2.imread(path)
    chImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, c, d = chImg.shape

    def openImage(imagePath):
        imOrig = Image.open(imagePath)
        im = np.array(imOrig)

        aRed = im[:, :, 0]
        aGreen = im[:, :, 1]
        aBlue = im[:, :, 2]

        return [aRed, aGreen, aBlue, imOrig]

    aRed, aGreen, aBlue, d = openImage(path)

    def compressSingleChannel(channelDataMatrix, singularValuesLimit):
        uChannel, sChannel, vhChannel = np.linalg.svd(channelDataMatrix)
        aChannelCompressed = np.zeros(
            (channelDataMatrix.shape[0], channelDataMatrix.shape[1]))
        k = singularValuesLimit

        leftSide = np.matmul(uChannel[:, 0:k], np.diag(sChannel)[0:k, 0:k])
        aChannelCompressedInner = np.matmul(leftSide, vhChannel[0:k, :])
        aChannelCompressed = aChannelCompressedInner.astype('uint8')

        return aChannelCompressed

    singularValuesLimit = sv

    aRedCompressed = compressSingleChannel(aRed, singularValuesLimit)
    aGreenCompressed = compressSingleChannel(aGreen, singularValuesLimit)
    aBlueCompressed = compressSingleChannel(aBlue, singularValuesLimit)

    imr = Image.fromarray(aRedCompressed, mode=None)
    img = Image.fromarray(aGreenCompressed, mode=None)
    imb = Image.fromarray(aBlueCompressed, mode=None)

    newImage = Image.merge("RGB", (imr, img, imb))

    file = request.files['file']
    name_image = file.filename
    result_image = name_image + '_' + str(sv) + '_SVD_compress.png'
    newImage.save(os.path.join(RESULT_FOLDER, result_image))

    # Calculate MSE
    i = 0
    j = 0
    sq_error = 0
    while i < r:
        while j < c:
            sq_error += (aRed[i][j] - aRedCompressed[i][j])**2
            sq_error += (aGreen[i][j] - aGreenCompressed[i][j])**2
            sq_error += (aBlue[i][j] - aBlueCompressed[i][j])**2
            j += 1
        j = 0
        i += 1

    MSE = sq_error/(3*r*c)

    # Calculate PSNR
    PSNR = np.log10(255*765/abs(MSE))

    # Calculate CR
    original_size = os.path.getsize(UPLOAD_FOLDER + name_image)
    compressed_size = os.path.getsize(RESULT_FOLDER + result_image)
    CR = original_size/compressed_size

    compress = os.path.join(RESULT_FOLDER, result_image)

    MSE = round(MSE, 5)
    PSNR = round(PSNR, 5)
    CR = round(CR, 5)

    return abs(MSE), PSNR, CR, compress


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@ app.route('/')
def home():
    return render_template('index.html')


@ app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No File Part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No Image Selected for Uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image Sucessfully Uploaded and Displayed Below')

        algorithm = request.form.get('algorithm')
        par1 = int(request.form.get('par1'))
        par2 = int(request.form.get('par2'))
        par3 = int(request.form.get('par3'))
        par4 = int(request.form.get('par4'))
        image = UPLOAD_FOLDER + filename

        if algorithm == 'DCT':
            par_dct = []
            mse_dct = []
            psnr_dct = []
            cr_dct = []

            mse1, psnr1, cr1, compress1 = dct_algorithm(image, 8, par1)
            par_dct.append(par1)
            mse_dct.append(mse1)
            psnr_dct.append(psnr1)
            cr_dct.append(cr1)

            mse2, psnr2, cr2, compress2 = dct_algorithm(image, 8, par2)
            par_dct.append(par2)
            mse_dct.append(mse2)
            psnr_dct.append(psnr2)
            cr_dct.append(cr2)

            mse3, psnr3, cr3, compress3 = dct_algorithm(image, 8, par3)
            par_dct.append(par3)
            mse_dct.append(mse3)
            psnr_dct.append(psnr3)
            cr_dct.append(cr3)

            mse4, psnr4, cr4, compress4 = dct_algorithm(image, 8, par4)
            par_dct.append(par4)
            mse_dct.append(mse4)
            psnr_dct.append(psnr4)
            cr_dct.append(cr4)

            df = pd.DataFrame({'Quality(%)': par_dct, 'MSE': mse_dct,
                               'PSNR': psnr_dct, 'Compression Ratio': cr_dct})
            csv = filename + '_dct.csv'
            df.to_csv(os.path.join(STAT_FOLDER, csv))

            mse_plot = sns.lineplot(x='Quality(%)', y='MSE',
                                    color='aqua', marker='o', data=df)
            fig1 = mse_plot.get_figure()
            plot1 = 'MSE' + filename + '_dct.png'
            fig1.savefig(os.path.join(STAT_FOLDER, plot1))
            mse_plot.get_figure().clf()
            stat1 = os.path.join(STAT_FOLDER, plot1)

            psnr_plot = sns.lineplot(x='Quality(%)', y='PSNR',
                                     color='red', marker='o', data=df)
            fig2 = psnr_plot.get_figure()
            plot2 = 'PSNR' + filename + '_dct.png'
            fig2.savefig(os.path.join(STAT_FOLDER, plot2))
            psnr_plot.get_figure().clf()
            stat2 = os.path.join(STAT_FOLDER, plot2)

            cr_plot = sns.lineplot(x='Quality(%)', y='Compression Ratio',
                                   color='green', marker='o', data=df)
            fig3 = cr_plot.get_figure()
            plot3 = 'CR' + filename + '_dct.png'
            fig3.savefig(os.path.join(STAT_FOLDER, plot3))
            cr_plot.get_figure().clf()
            stat3 = os.path.join(STAT_FOLDER, plot3)

        elif algorithm == 'DWT':
            par_dwt = []
            mse_dwt = []
            psnr_dwt = []
            cr_dwt = []

            mse1, psnr1, cr1, compress1 = dwt_algorithm(image, par1)
            par_dwt.append(par1)
            mse_dwt.append(mse1)
            psnr_dwt.append(psnr1)
            cr_dwt.append(cr1)

            mse2, psnr2, cr2, compress2 = dwt_algorithm(image, par2)
            par_dwt.append(par2)
            mse_dwt.append(mse2)
            psnr_dwt.append(psnr2)
            cr_dwt.append(cr2)

            mse3, psnr3, cr3, compress3 = dwt_algorithm(image, par3)
            par_dwt.append(par3)
            mse_dwt.append(mse3)
            psnr_dwt.append(psnr3)
            cr_dwt.append(cr3)

            mse4, psnr4, cr4, compress4 = dwt_algorithm(image, par4)
            par_dwt.append(par4)
            mse_dwt.append(mse4)
            psnr_dwt.append(psnr4)
            cr_dwt.append(cr4)

            df = pd.DataFrame({'Quality(%)': par_dwt, 'MSE': mse_dwt,
                               'PSNR': psnr_dwt, 'Compression Ratio': cr_dwt})
            csv = filename + '_dwt.csv'
            df.to_csv(os.path.join(STAT_FOLDER, csv))

            mse_plot = sns.lineplot(x='Quality(%)', y='MSE',
                                    color='aqua', marker='o', data=df)
            fig1 = mse_plot.get_figure()
            plot1 = 'MSE' + filename + '_dwt.png'
            fig1.savefig(os.path.join(STAT_FOLDER, plot1))
            mse_plot.get_figure().clf()
            stat1 = os.path.join(STAT_FOLDER, plot1)

            psnr_plot = sns.lineplot(x='Quality(%)', y='PSNR',
                                     color='red', marker='o', data=df)
            fig2 = psnr_plot.get_figure()
            plot2 = 'PSNR' + filename + '_dwt.png'
            fig2.savefig(os.path.join(STAT_FOLDER, plot2))
            psnr_plot.get_figure().clf()
            stat2 = os.path.join(STAT_FOLDER, plot2)

            cr_plot = sns.lineplot(x='Quality(%)', y='Compression Ratio',
                                   color='green', marker='o', data=df)
            fig3 = cr_plot.get_figure()
            plot3 = 'CR' + filename + '_dwt.png'
            fig3.savefig(os.path.join(STAT_FOLDER, plot3))
            cr_plot.get_figure().clf()
            stat3 = os.path.join(STAT_FOLDER, plot3)

        elif algorithm == 'BTC':
            par_btc = []
            mse_btc = []
            psnr_btc = []
            cr_btc = []

            mse1, psnr1, cr1, compress1 = btc_algorithm(image, par1)
            par_btc.append(par1)
            mse_btc.append(mse1)
            psnr_btc.append(psnr1)
            cr_btc.append(cr1)

            mse2, psnr2, cr2, compress2 = btc_algorithm(image, par2)
            par_btc.append(par2)
            mse_btc.append(mse2)
            psnr_btc.append(psnr2)
            cr_btc.append(cr2)

            mse3, psnr3, cr3, compress3 = btc_algorithm(image, par3)
            par_btc.append(par3)
            mse_btc.append(mse3)
            psnr_btc.append(psnr3)
            cr_btc.append(cr3)

            mse4, psnr4, cr4, compress4 = btc_algorithm(image, par4)
            par_btc.append(par4)
            mse_btc.append(mse4)
            psnr_btc.append(psnr4)
            cr_btc.append(cr4)

            df = pd.DataFrame({'Quality(%)': par_btc, 'MSE': mse_btc,
                               'PSNR': psnr_btc, 'Compression Ratio': cr_btc})
            csv = filename + '_btc.csv'
            df.to_csv(os.path.join(STAT_FOLDER, csv))

            mse_plot = sns.lineplot(x='Quality(%)', y='MSE',
                                    color='aqua', marker='o', data=df)
            fig1 = mse_plot.get_figure()
            plot1 = 'MSE' + filename + '_btc.png'
            fig1.savefig(os.path.join(STAT_FOLDER, plot1))
            mse_plot.get_figure().clf()
            stat1 = os.path.join(STAT_FOLDER, plot1)

            psnr_plot = sns.lineplot(x='Quality(%)', y='PSNR',
                                     color='red', marker='o', data=df)
            fig2 = psnr_plot.get_figure()
            plot2 = 'PSNR' + filename + '_btc.png'
            fig2.savefig(os.path.join(STAT_FOLDER, plot2))
            psnr_plot.get_figure().clf()
            stat2 = os.path.join(STAT_FOLDER, plot2)

            cr_plot = sns.lineplot(x='Quality(%)', y='Compression Ratio',
                                   color='green', marker='o', data=df)
            fig3 = cr_plot.get_figure()
            plot3 = 'CR' + filename + '_btc.png'
            fig3.savefig(os.path.join(STAT_FOLDER, plot3))
            cr_plot.get_figure().clf()
            stat3 = os.path.join(STAT_FOLDER, plot3)

        elif algorithm == 'AMBTC':
            par_ambtc = []
            mse_ambtc = []
            psnr_ambtc = []
            cr_ambtc = []

            mse1, psnr1, cr1, compress1 = ambtc_algorithm(image, 4, par1)
            par_ambtc.append(par1)
            mse_ambtc.append(mse1)
            psnr_ambtc.append(psnr1)
            cr_ambtc.append(cr1)

            mse2, psnr2, cr2, compress2 = ambtc_algorithm(image, 4, par2)
            par_ambtc.append(par2)
            mse_ambtc.append(mse2)
            psnr_ambtc.append(psnr2)
            cr_ambtc.append(cr2)

            mse3, psnr3, cr3, compress3 = ambtc_algorithm(image, 4, par3)
            par_ambtc.append(par3)
            mse_ambtc.append(mse3)
            psnr_ambtc.append(psnr3)
            cr_ambtc.append(cr3)

            mse4, psnr4, cr4, compress4 = ambtc_algorithm(image, 4, par4)
            par_ambtc.append(par4)
            mse_ambtc.append(mse4)
            psnr_ambtc.append(psnr4)
            cr_ambtc.append(cr4)

            df = pd.DataFrame({'Quality(%)': par_ambtc, 'MSE': mse_ambtc,
                               'PSNR': psnr_ambtc, 'Compression Ratio': cr_ambtc})
            csv = filename + '_ambtc.csv'
            df.to_csv(os.path.join(STAT_FOLDER, csv))

            mse_plot = sns.lineplot(x='Quality(%)', y='MSE',
                                    color='aqua', marker='o', data=df)
            fig1 = mse_plot.get_figure()
            plot1 = 'MSE' + filename + '_ambtc.png'
            fig1.savefig(os.path.join(STAT_FOLDER, plot1))
            mse_plot.get_figure().clf()
            stat1 = os.path.join(STAT_FOLDER, plot1)

            psnr_plot = sns.lineplot(x='Quality(%)', y='PSNR',
                                     color='red', marker='o', data=df)
            fig2 = psnr_plot.get_figure()
            plot2 = 'PSNR' + filename + '_ambtc.png'
            fig2.savefig(os.path.join(STAT_FOLDER, plot2))
            psnr_plot.get_figure().clf()
            stat2 = os.path.join(STAT_FOLDER, plot2)

            cr_plot = sns.lineplot(x='Quality(%)', y='Compression Ratio',
                                   color='green', marker='o', data=df)
            fig3 = cr_plot.get_figure()
            plot3 = 'CR' + filename + '_ambtc.png'
            fig3.savefig(os.path.join(STAT_FOLDER, plot3))
            cr_plot.get_figure().clf()
            stat3 = os.path.join(STAT_FOLDER, plot3)

        elif algorithm == 'SVD':
            par_svd = []
            mse_svd = []
            psnr_svd = []
            cr_svd = []

            mse1, psnr1, cr1, compress1 = svd_algorithm(image, par1)
            par_svd.append(par1)
            mse_svd.append(mse1)
            psnr_svd.append(psnr1)
            cr_svd.append(cr1)

            mse2, psnr2, cr2, compress2 = svd_algorithm(image, par2)
            par_svd.append(par2)
            mse_svd.append(mse2)
            psnr_svd.append(psnr2)
            cr_svd.append(cr2)

            mse3, psnr3, cr3, compress3 = svd_algorithm(image, par3)
            par_svd.append(par3)
            mse_svd.append(mse3)
            psnr_svd.append(psnr3)
            cr_svd.append(cr3)

            mse4, psnr4, cr4, compress4 = svd_algorithm(image, par4)
            par_svd.append(par4)
            mse_svd.append(mse4)
            psnr_svd.append(psnr4)
            cr_svd.append(cr4)

            df = pd.DataFrame({'Quality(%)': par_svd, 'MSE': mse_svd,
                               'PSNR': psnr_svd, 'Compression Ratio': cr_svd})
            csv = filename + '_svd.csv'
            df.to_csv(os.path.join(STAT_FOLDER, csv))

            mse_plot = sns.lineplot(x='Quality(%)', y='MSE',
                                    color='aqua', marker='o', data=df)
            fig1 = mse_plot.get_figure()
            plot1 = 'MSE' + filename + '_svd.png'
            fig1.savefig(os.path.join(STAT_FOLDER, plot1))
            mse_plot.get_figure().clf()
            stat1 = os.path.join(STAT_FOLDER, plot1)

            psnr_plot = sns.lineplot(x='Quality(%)', y='PSNR',
                                     color='red', marker='o', data=df)
            fig2 = psnr_plot.get_figure()
            plot2 = 'PSNR' + filename + '_svd.png'
            fig2.savefig(os.path.join(STAT_FOLDER, plot2))
            psnr_plot.get_figure().clf()
            stat2 = os.path.join(STAT_FOLDER, plot2)

            cr_plot = sns.lineplot(x='Quality(%)', y='Compression Ratio',
                                   color='green', marker='o', data=df)
            fig3 = cr_plot.get_figure()
            plot3 = 'CR' + filename + '_svd.png'
            fig3.savefig(os.path.join(STAT_FOLDER, plot3))
            cr_plot.get_figure().clf()
            stat3 = os.path.join(STAT_FOLDER, plot3)

        return render_template('index.html', filename=filename,
                               compress1=compress1, compress2=compress2, compress3=compress3, compress4=compress4,
                               mse1=mse1, psnr1=psnr1, cr1=cr1,
                               mse2=mse2, psnr2=psnr2, cr2=cr2,
                               mse3=mse3, psnr3=psnr3, cr3=cr3,
                               mse4=mse4, psnr4=psnr4, cr4=cr4,
                               stat1=stat1, stat2=stat2, stat3=stat3)

    else:
        flash('Allowed Image types Bitmap (.bmp)')
        return redirect(request.url)


@ app.route('/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run(debug=True)
