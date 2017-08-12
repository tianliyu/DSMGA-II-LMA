
#ifndef _STATISTICS_H
#define _STATISTICS_H

#include <cmath>
#include <climits>

/**
@author Tian-Li Yu
*/

#ifndef INF
#define INF (1e10)
#endif

class Statistics {

public:

    Statistics () {
        reset ();
    }


    void reset () {
        precision = 1e-6;
        min = INF;
        second_min = INF;
        max = -INF;
        second_max = -INF;
        sum = 0.0;
        variance = 0.0;
        number = 0;
        status = true;
    }

    void record (double value) {

        if (status == false)
            return;

        number++;
        sum += value;
        variance += value * value;
        if (min > value + precision) {
            second_min = min;
            min = value;
        }
        if (max < value - precision) {
            second_max = max;
            max = value;
        }
    }

    /** get the number of samples */
    long int getNumber () {
        return number;
    }

    /** get mean */
    double getMean () {
        return sum / number;
    }

    /** get variance */
    double getVariance () {
        double mean = getMean ();
        return variance / number - mean * mean;
    }

    /** get standard deviation */
    double getStdev () {
        return::sqrt (getVariance ());
    }

    double getMin () {
        return min;
    }

    double getMax () {
        return max;
    }

    double getSecondMax () {
        return second_max;
    }

    double getSecondMin () {
        return second_min;
    }

    void turnOn () {
        status = true;
    }

    void turnOff () {
        status = false;
    }

private:

    double precision;
    double min;
    double max;
    double second_min;
    double second_max;
    double sum;
    double variance;
    long int number;
    bool status;
};

#endif
