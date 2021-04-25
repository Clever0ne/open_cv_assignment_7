#pragma once

#include <cstdint>

#include "opencv2/core.hpp"

enum class Stage
{
    STAGE_ONE = 0,
    STAGE_TWO = 1
};

bool thinnig(cv::Mat &inputImage, cv::Mat &outputImage);

uint32_t countWhitePixels(const uint8_t *p0, const uint8_t *p1, const uint8_t *p2);
uint32_t countTransitions(const uint8_t *p0, const uint8_t *p1, const uint8_t *p2);
bool areBorderPixelsBlack(const uint8_t *p0, const uint8_t *p1, const uint8_t *p2, Stage stage);

bool drawPath(cv::Mat &inputImage, cv::Mat &outputImage);

bool findCoins(cv::Mat &inputImage, cv::Mat &outputImage, cv::Mat &nickelTemplate, cv::Mat &copperTemplate);

char waitForAnswer(const cv::String &question);
