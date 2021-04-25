#include "functions.h"

#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

bool thinnig(Mat &inputImage, Mat &outputImage)
{
    if (inputImage.empty() != false)
    {
        return false;
    }

    auto image = inputImage.clone();
    image /= 0xFF;

    auto isImageChanged = true;
    while (isImageChanged != false)
    {
        isImageChanged = false;
        static auto stage = Stage::STAGE_ONE;

        auto buffer = image.clone();
        auto p0 = buffer.ptr<uint8_t>(0);
        auto p1 = buffer.ptr<uint8_t>(1);

        for (auto row = 1; row < buffer.rows - 1; row++)
        {
            auto p = image.ptr<uint8_t>(row);
            auto p2 = buffer.ptr<uint8_t>(row + 1);

            for (auto col = 1; col < buffer.cols - 1; col++)
            {
                // Если пиксель не является белым,
                // переходим к следующему пикселю
                if (p1[col] != 1)
                {
                    continue;
                }

                // Если количество белых пикселей не лежит в диапазоне [2; 6],
                // переходим к следующему пикселю
                auto whitePixels = countWhitePixels(p0 + col - 1, p1 + col - 1, p2 + col - 1);
                if (whitePixels < 2 || whitePixels > 6)
                {
                    continue;
                }

                // Если количество переходов от чёрного пикселя к белому не равно 1,
                // переходим к следующему пикселю
                auto transitions = countTransitions(p0 + col - 1, p1 + col - 1, p2 + col - 1);
                if (transitions != 1)
                {
                    continue;
                }

                auto areBlack = areBorderPixelsBlack(p0 + col - 1, p1 + col - 1, p2 + col - 1, stage);
                if (areBlack != true)
                {
                    continue;
                }

                p[col] = 0;
                isImageChanged = true;
            }

            p0 = p1;
            p1 = p2;
        }

        stage == Stage::STAGE_ONE ? stage = Stage::STAGE_TWO : stage = Stage::STAGE_ONE;
    }

    outputImage = image * 0xFF;
    return true;
}

uint32_t countWhitePixels(const uint8_t *p0, const uint8_t *p1, const uint8_t *p2)
{
    auto counter = 0;
    counter += p0[0] + p0[1] + p0[2];
    counter += p1[0] + p1[2];
    counter += p2[0] + p2[1] + p2[2];
    return counter;
}

uint32_t countTransitions(const uint8_t *p0, const uint8_t *p1, const uint8_t *p2)
{
    auto counter = 0;
    counter += ~p0[1] & p0[2];
    counter += ~p0[2] & p1[2];
    counter += ~p1[2] & p2[2];
    counter += ~p2[2] & p2[1];
    counter += ~p2[1] & p2[0];
    counter += ~p2[0] & p1[0];
    counter += ~p1[0] & p0[0];
    counter += ~p0[0] & p0[1];
    return counter;
}

bool areBorderPixelsBlack(const uint8_t *p0, const uint8_t *p1, const uint8_t *p2, const Stage stage)
{
    switch (stage)
    {
        case Stage::STAGE_ONE:
        {
            return !static_cast<bool>(p1[2] & p2[1] & (p0[1] | p1[0]));
        }

        case Stage::STAGE_TWO:
        {
            return !static_cast<bool>(p0[1] & p1[0] & (p1[2] | p2[1]));
        }
    }

    return false;
}

bool drawPath(cv::Mat &inputImage, cv::Mat &outputImage)
{
    if (inputImage.empty() != false)
    {
        return false;
    }

    auto frame = inputImage.clone();
    auto result = Mat();
    auto grayImage = Mat();

    cvtColor(frame, grayImage, COLOR_RGB2GRAY);
    grayImage.convertTo(grayImage, CV_8U);
    GaussianBlur(grayImage, grayImage, Size(5, 5), 1, 1);
    threshold(grayImage, grayImage, 0x4F, 0xFF, THRESH_BINARY);

    auto kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(grayImage, grayImage, MORPH_CLOSE, kernel);
    morphologyEx(grayImage, grayImage, MORPH_OPEN, kernel);
    thinnig(grayImage, result);

    auto cols = result.cols;
    auto rows = result.rows;

    line(result, Point(0, 0), Point(0, rows - 1), Scalar(0x00));
    line(result, Point(0, 0), Point(cols - 1, 0), Scalar(0x00));
    line(result, Point(cols - 1, rows - 1), Point(0, rows - 1), Scalar(0x00));
    line(result, Point(cols - 1, rows - 1), Point(cols - 1, 0), Scalar(0x00));
    rectangle(result, Point(0, 0), Point(cols - 1, rows / 2 - 1), Scalar(0x00), -1, LINE_8, 0);

    auto pathLines = vector<Vec4i>();
    HoughLinesP(result, pathLines, 1, CV_PI / 180, 50, 10, 50);
    for (auto& pathLine : pathLines)
    {
        line(frame, Point(pathLine[0], pathLine[1]), Point(pathLine[2], pathLine[3]), Scalar(0x00, 0xFF, 0x00), 2);
    }

    imshow("Threshold", grayImage);
    imshow("Thinning", result);

    outputImage = frame;

    return true;
}

bool findCoins(cv::Mat &inputImage, cv::Mat &outputImage, cv::Mat &nickelTemplate, cv::Mat &copperTemplate)
{
    if (inputImage.empty() != false)
    {
        return false;
    }

    if (nickelTemplate.empty() != false)
    {
        return false;
    }

    if (copperTemplate.empty() != false)
    {
        return false;
    }

    auto image = inputImage.clone();
    auto grayImage = Mat();

    cvtColor(image, grayImage, COLOR_RGB2GRAY);
    grayImage.convertTo(grayImage, CV_8U);
    GaussianBlur(grayImage, grayImage, Size(5, 5), 1, 1);

    auto circles = vector<Vec3f>();
    HoughCircles(grayImage, circles, HOUGH_GRADIENT, 1, grayImage.rows / 7, 100, 30, 10, 100);

    auto coins = vector<Mat>();
    for (auto &circle : circles)
    {
        auto center = Point(static_cast<uint32_t>(circle[0]), 
                            static_cast<uint32_t>(circle[1]));
        auto radius = static_cast<uint32_t>(circle[2]);

        auto rectangle = Rect(center.x - radius / 2, center.y - radius / 2, radius, radius);
        coins.emplace_back(image, rectangle);
    }

    auto hsvCoins = vector<Mat>();
    for (auto &coin : coins)
    {
        auto hsvCoin = Mat();
        cvtColor(coin, hsvCoin, COLOR_RGB2HSV);
        hsvCoins.emplace_back(hsvCoin);
    }

    auto hsvNickelCoin = Mat();
    cvtColor(nickelTemplate, hsvNickelCoin, COLOR_RGB2HSV);

    auto hsvCopperCoin = Mat();
    cvtColor(copperTemplate, hsvCopperCoin, COLOR_RGB2HSV);

    auto hue = 180;
    auto sat = 256;
    auto val = 256;
    int32_t histSize[] = { hue, sat, val };

    float hueRanges[] = { 0, hue - 1 };
    float satRanges[] = { 0, sat - 1 };
    float valRanges[] = { 0, val - 1 };
    const float *ranges[] = { hueRanges, satRanges, valRanges };

    int channels[] = { 0, 1, 2 };

    auto coinHists = vector<Mat>();
    for (auto &hsvCoin : hsvCoins)
    {
        auto coinHist = Mat();
        calcHist(&hsvCoin, 1, channels, Mat(), coinHist, 2, histSize, ranges);
        normalize(coinHist, coinHist, 0, 1, NORM_MINMAX, -1, Mat());
        coinHists.emplace_back(coinHist);
    }

    auto nickelCoinHist = Mat();
    auto copperCoinHist = Mat();

    calcHist(&hsvNickelCoin, 1, channels, Mat(), nickelCoinHist, 2, histSize, ranges);
    normalize(nickelCoinHist, nickelCoinHist, 0, 1, NORM_MINMAX, -1, Mat());

    calcHist(&hsvCopperCoin, 1, channels, Mat(), copperCoinHist, 2, histSize, ranges);
    normalize(copperCoinHist, copperCoinHist, 0, 1, NORM_MINMAX, -1, Mat());

    auto coinType = vector<bool>();
    for (auto &coinHist : coinHists)
    {
        auto compareWithNickelCoin = compareHist(coinHist, nickelCoinHist, HISTCMP_CORREL);
        auto compareWithCopperCoin = compareHist(coinHist, copperCoinHist, HISTCMP_CORREL);

        if (compareWithNickelCoin > compareWithCopperCoin)
        {
            coinType.emplace_back(true);
        }
        else
        {
            coinType.emplace_back(false);
        }
    }

    for (auto i = 0; i < coinType.size(); i++)
    {
        if (coinType.at(i) == true)
        {
            auto center = Point(static_cast<uint32_t>(circles.at(i)[0]), 
                                static_cast<uint32_t>(circles.at(i)[1]));
            auto radius = static_cast<uint32_t>(circles.at(i)[2]);
            circle(image, center, radius, Scalar(0, 0xFF, 0), 2, LINE_AA);
        }
        else
        {
            auto center = Point(static_cast<uint32_t>(circles.at(i)[0]), 
                                static_cast<uint32_t>(circles.at(i)[1]));
            auto radius = static_cast<uint32_t>(circles.at(i)[2]);
            circle(image, center, radius, Scalar(0xFF, 0, 0), 2, LINE_AA);
        }
    }

    outputImage = image;

    return true;
}

char waitForAnswer(const String &question)
{
    cout << endl << question;

    char key = -1;
    while (key != 'y' && key != 'n')
    {
        if (key != -1)
        {
            cout << "Please, enter your answer! ";
        }
        cin >> key;
    }

    return key;
}
