#include <cstdio>
#include <arm_neon.h>
#include "neon_mathfun.h"
#include <iostream>
#include <random>


#define inv_sqrt_2xPI 0.39894228040143270286

float32x4_t CNDF(float32x4_t InputX) {
    const float32x4_t vInvSqrt2xPI = vdupq_n_f32(inv_sqrt_2xPI);

    float32x4_t vSign = vcltq_f32(InputX, vdupq_n_f32(0.0));
    float32x4_t vInputX = vabsq_f32(InputX);

    float32x4_t xK2, xK2_2, xK2_3, xK2_4, xK2_5;
    float32x4_t xLocal, xLocal_1, xLocal_2, xLocal_3;

    float32x4_t vNPrimeofX = exp_ps(
            vmulq_f32(
                    vdupq_n_f32(-0.5),
                    vmulq_f32(vInputX, vInputX)
            )
    );
    vNPrimeofX = vmulq_f32(vNPrimeofX, vInvSqrt2xPI);

    xK2 = vmulq_f32(vdupq_n_f32(0.2316419), vInputX);
    xK2 = vaddq_f32(vdupq_n_f32(1.0), xK2);
    xK2 = vdivq_f32(vdupq_n_f32(1.0), xK2);
    xK2_2 = vmulq_f32(xK2, xK2);
    xK2_3 = vmulq_f32(xK2_2, xK2);
    xK2_4 = vmulq_f32(xK2_3, xK2);
    xK2_5 = vmulq_f32(xK2_4, xK2);

    xLocal_1 = vmulq_f32(xK2, vdupq_n_f32(0.319381530));
    xLocal_2 = vmulq_f32(xK2_2, vdupq_n_f32(-0.356563782));
    xLocal_3 = vmulq_f32(xK2_3, vdupq_n_f32(1.781477937));
    xLocal_2 = vaddq_f32(xLocal_2, xLocal_3);
    xLocal_3 = vmulq_f32(xK2_4, vdupq_n_f32(-1.821255978));
    xLocal_2 = vaddq_f32(xLocal_2, xLocal_3);
    xLocal_3 = vmulq_f32(xK2_5, vdupq_n_f32(1.330274429));
    xLocal_2 = vaddq_f32(xLocal_2, xLocal_3);

    xLocal_1 = vaddq_f32(xLocal_2, xLocal_1);
    xLocal = vmulq_f32(xLocal_1, vNPrimeofX);
    xLocal = vsubq_f32(vdupq_n_f32(1.0), xLocal);

    float32x4_t signedOutput = vsubq_f32(vdupq_n_f32(1.0), xLocal);

    return vbslq_f32(vSign, signedOutput, xLocal);;
}


float32x4_t
blackScholes(float32x4_t vSptPrice, float32x4_t vStrike,
             float32x4_t vRate, float32x4_t vVolatility,
             float32x4_t vOtime, int32x4_t vOType) {

    float32x4_t vSqrtTime = vsqrtq_f32(vOtime);
    float32x4_t vLogTerm = log_ps(
            vdivq_f32(vSptPrice, vStrike)
    );

    float32x4_t vPowerTerm = vmulq_f32(vVolatility, vVolatility);
    vPowerTerm = vmulq_f32(vPowerTerm, vdupq_n_f32(0.5));

    float32x4_t xD1 = vaddq_f32(vRate, vPowerTerm);
    xD1 = vmulq_f32(xD1, vOtime);
    xD1 = vaddq_f32(xD1, vLogTerm);

    float32x4_t vDen = vmulq_f32(vVolatility, vSqrtTime);
    xD1 = vdivq_f32(xD1, vDen);
    float32x4_t xD2 = vsubq_f32(xD1, vDen);

    float32x4_t vCDFd1 = CNDF(xD1);
    float32x4_t vCDFd2 = CNDF(xD2);

    float32x4_t vFutureValueX = vmulq_f32(
            vStrike,
            exp_ps(
                    vmulq_f32(
                            vmulq_f32(vdupq_n_f32(-1), vRate),
                            vOtime
                    )
            )
    );

    float32x4_t vOptionPriceOTypeC = vsubq_f32(
            vmulq_f32(vSptPrice, vCDFd1),
            vmulq_f32(vFutureValueX, vCDFd2)
    );

    float32x4_t vOptionPriceOTypeP = vsubq_f32(
            vmulq_f32(vFutureValueX, vsubq_f32(vdupq_n_f32(1), vCDFd2)),
            vmulq_f32(vSptPrice, vsubq_f32(vdupq_n_f32(1), vCDFd1))
    );

    float32x4_t vOptionPrice = vbslq_f32(
            vceqzq_f32(vOType),
            vOptionPriceOTypeC,
            vOptionPriceOTypeP
    );

    return vOptionPrice;
}

void generateBlackScholesCases(float spotPriceArr[], float strikeArr[], float rateArr[],
                               float volatilityArr[], float timeArr[], int typeArr[], int size) {
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator

    // Define distributions for each parameter
    std::uniform_real_distribution<float> spotPriceDist(30.0, 150.0);
    std::uniform_real_distribution<float> strikePriceDist(30.0, 150.0);
    std::uniform_real_distribution<float> rateDist(0.01, 0.1);
    std::uniform_real_distribution<float> volatilityDist(0.1, 0.5);
    std::uniform_real_distribution<float> timeDist(0.1, 2.0);
    std::uniform_int_distribution<int> typeDist(0, 1);

    // Generate data
    for (int i = 0; i < size; ++i) {
        spotPriceArr[i] = spotPriceDist(gen);
        strikeArr[i] = strikePriceDist(gen);
        rateArr[i] = rateDist(gen);
        volatilityArr[i] = volatilityDist(gen);
        timeArr[i] = timeDist(gen);
        typeArr[i] = typeDist(gen);
    }
}

float CNDFSISD(float InputX) {
    int sign;

    float OutputX;
    float xInput;
    float xNPrimeofX;
    float expValues;
    float xK2;
    float xK2_2, xK2_3;
    float xK2_4, xK2_5;
    float xLocal, xLocal_1;
    float xLocal_2, xLocal_3;

    // Check for negative value of InputX
    if (InputX < 0.0) {
        InputX = -InputX;
        sign = 1;
    } else
        sign = 0;

    xInput = InputX;

    // Compute NPrimeX term common to both four & six decimal accuracy calcs
    expValues = exp(-0.5f * InputX * InputX);
    xNPrimeofX = expValues;
    xNPrimeofX = xNPrimeofX * inv_sqrt_2xPI;

    xK2 = 0.2316419 * xInput;
    xK2 = 1.0 + xK2;
    xK2 = 1.0 / xK2;
    xK2_2 = xK2 * xK2;
    xK2_3 = xK2_2 * xK2;
    xK2_4 = xK2_3 * xK2;
    xK2_5 = xK2_4 * xK2;

    xLocal_1 = xK2 * 0.319381530;
    xLocal_2 = xK2_2 * (-0.356563782);
    xLocal_3 = xK2_3 * 1.781477937;
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_4 * (-1.821255978);
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_5 * 1.330274429;
    xLocal_2 = xLocal_2 + xLocal_3;

    xLocal_1 = xLocal_2 + xLocal_1;
    xLocal = xLocal_1 * xNPrimeofX;
    xLocal = 1.0 - xLocal;

    OutputX = xLocal;

    if (sign) {
        OutputX = 1.0 - OutputX;
    }

    return OutputX;
}


float blackScholesSISD(float sptprice, float strike, float rate, float volatility,
                       float otime, int otype, float timet) {
    float OptionPrice;

    // local private working variables for the calculation
    float xStockPrice;
    float xStrikePrice;
    float xRiskFreeRate;
    float xVolatility;
    float xTime;
    float xSqrtTime;

    float logValues;
    float xLogTerm;
    float xD1;
    float xD2;
    float xPowerTerm;
    float xDen;
    float d1;
    float d2;
    float FutureValueX;
    float NofXd1;
    float NofXd2;
    float NegNofXd1;
    float NegNofXd2;

    xStockPrice = sptprice;
    xStrikePrice = strike;
    xRiskFreeRate = rate;
    xVolatility = volatility;

    xTime = otime;
    xSqrtTime = sqrt(xTime);

    logValues = log(sptprice / strike);

    xLogTerm = logValues;


    xPowerTerm = xVolatility * xVolatility;
    xPowerTerm = xPowerTerm * 0.5;

    xD1 = xRiskFreeRate + xPowerTerm;
    xD1 = xD1 * xTime;
    xD1 = xD1 + xLogTerm;

    xDen = xVolatility * xSqrtTime;
    xD1 = xD1 / xDen;
    xD2 = xD1 - xDen;

    d1 = xD1;
    d2 = xD2;

    NofXd1 = CNDFSISD(d1);
    NofXd2 = CNDFSISD(d2);

    FutureValueX = strike * (exp(-(rate) * (otime)));
    if (otype == 0) {
        OptionPrice = (sptprice * NofXd1) - (FutureValueX * NofXd2);
    } else {
        NegNofXd1 = (1.0 - NofXd1);
        NegNofXd2 = (1.0 - NofXd2);
        OptionPrice = (FutureValueX * NegNofXd2) - (sptprice * NegNofXd1);
    }

    return OptionPrice;
}

int main(int argc, char *argv[]) {

    // Number of test cases
    int dataSize = 65536;

    float spotPriceArr[dataSize], strikeArr[dataSize], rateArr[dataSize];
    float volatilityArr[dataSize], timeArr[dataSize];
    int typeArr[dataSize];

    // Generate test cases
    generateBlackScholesCases(spotPriceArr, strikeArr, rateArr, volatilityArr, timeArr, typeArr, dataSize);

    float OptionPrice[dataSize];
    float OptionPriceSISD[dataSize];

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < dataSize; i += 4) {
        float32x4_t vSpotPrice = vld1q_f32(&spotPriceArr[i]);
        float32x4_t vStrike = vld1q_f32(&strikeArr[i]);
        float32x4_t vRate = vld1q_f32(&rateArr[i]);
        float32x4_t vVolatility = vld1q_f32(&volatilityArr[i]);
        float32x4_t vOtime = vld1q_f32(&timeArr[i]);
        int32x4_t vOtype = vld1q_s32(&typeArr[i]);

        float32x4_t oprice = blackScholes(vSpotPrice, vStrike, vRate, vVolatility, vOtime, vOtype);

        vst1q_f32(&OptionPrice[i], oprice);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;

    std::cout << "Time taken using SIMD: " << diff.count() << " s\n";

    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < dataSize; i++) {
        OptionPriceSISD[i] = blackScholesSISD(spotPriceArr[i], strikeArr[i], rateArr[i], volatilityArr[i], timeArr[i],
                                              typeArr[i], 0);
    }

    end = std::chrono::high_resolution_clock::now();

    diff = end - start;

    std::cout << "Time taken using SISD: " << diff.count() << " s\n";

    return 0;
}
