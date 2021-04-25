#include <iostream>

#include "functions.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

static const String images_path = "src/images/";
static const String videos_path = "src/videos/";

int main(int argc, char *argv[])
{
    cout << "Starting a program.";

    /************** Задание 1. Скелетизация **************/

    auto question = "Would you like to test Zhang - Suen Thinning Algorithm [y/n]? ";
    auto key = waitForAnswer(question);

    if (key == 'y')
    {
        auto imageNames = vector<String>
        {
            "letter_a_small.jpg",
            "letter_a_big.jpg",
            "counter_strike.jpg"
        };

        for (auto &imageName : imageNames)
        {
            auto image = imread(images_path + imageName);
            auto result = Mat();

            cvtColor(image, image, COLOR_RGB2GRAY);
            image.convertTo(image, CV_8U);
            threshold(image, image, 0x7F, 0xFF, THRESH_BINARY);

            auto isOk = thinnig(image, result);
            if (isOk != true)
            {
                cout << "Something went wrong." << endl;
            }
            else
            {
                imshow("Original Image", image);
                imshow("Thinning", result);
            }

            while (waitKey() != 27);

            destroyAllWindows();
        }
    }

    /*********** Задание 2. Поиск прямых линий ***********/

    question = "Would you like to test applying the Hough Transform to find straight lines [y/n]? ";
    key = waitForAnswer(question);

    if (key == 'y')
    {
        auto videoNames = vector<String>
        {
            "video_1.avi",
            "video_2.avi",
            "video_3.avi",
            "video_4.avi"
        };

        for (auto &videoName : videoNames)
        {
            auto video = VideoCapture(videos_path + videoName);
            auto frame = Mat();

            auto isVideoEnd = false;
            while (isVideoEnd != true)
            {
                auto isOk = video.read(frame);
                if (isOk != true)
                {
                    isVideoEnd = true;
                    continue;
                }

                isOk = drawPath(frame, frame);
                if (isOk != false)
                {
                    imshow("Original Video", frame);
                }

                const auto key = static_cast<char>(waitKey(1));
                if (key == 'q' || key == 'Q' || key == 27)
                {
                    isVideoEnd = true;
                }
            }

            while (waitKey() != 27);

            destroyAllWindows();
        }
    }
    
    /************** Задание 3. Поиск кругов **************/

    question = "Would you like to test applying the Hough Transform to find circles [y/n]? ";
    key = waitForAnswer(question);

    if (key == 'y')
    {
        auto imageNames = vector<String>
        {
            "coins.jpg"
        };

        auto nickelTemplate = imread(images_path + "nickel_coin.jpg");
        auto copperTemplate = imread(images_path + "copper_coin.jpg");

        for (auto &imageName : imageNames)
        {
            auto image = imread(images_path + "coins.jpg");

            auto isOk = findCoins(image, image, nickelTemplate, copperTemplate);
            if (isOk != true)
            {
                cout << "Something went wrong." << endl;
            }
            else
            {
                imshow("Coins", image);
            }

            while (waitKey() != 27);

            destroyAllWindows();
        }
    }

    return 0;
}   
