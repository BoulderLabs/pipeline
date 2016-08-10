#ifndef LIQUIDATOR_DETAIL_SCORE_MATRIX_DETAIL_H_INCLUDED
#define LIQUIDATOR_DETAIL_SCORE_MATRIX_DETAIL_H_INCLUDED

#include "liquidator_util.h"
#include "pwm_detail.h"
#include "parsing_detail.h"

#include <iostream>
#include <map>
#include <string>
#include <bitset>

namespace liquidator { namespace detail {

struct ScaledPWM
{
    const size_t number_of_sites;
    const int min_before_scaling; // floored min before scaling
    const unsigned scale;
    const unsigned range;
    std::string name;

    // matrix values are scaled and offset by min so that they are between 0 and range.
    std::vector<std::array<unsigned, AlphabetSize>> matrix;
};

// Transforms PWM probability values into log pseudo-site-adjusted likelihood ratio values.
// Returns min/max ratio values.
inline std::pair<double, double>
log_adjusted_likelihood_ratio(PWM& pwm, 
                              const std::array<double, AlphabetSize>& original_background,
                              const std::array<double, AlphabetSize>& adjusted_background,
                              const double number_of_pseudo_sites=.1)
{
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    for (auto& row : pwm.matrix)
    {
        for (size_t i=0; i < AlphabetSize; ++i)
        {
            const double adjusted = (row[i]*pwm.number_of_sites + number_of_pseudo_sites*original_background[i])/(pwm.number_of_sites+number_of_pseudo_sites);
            const double ratio = adjusted / adjusted_background[i];
            const double log_ratio = std::log2(ratio);
            min = std::min(min, log_ratio);
            max = std::max(max, log_ratio);
            row[i] = log_ratio;
        }
    }
    return std::make_pair(min, max);
}

inline ScaledPWM
scale(const PWM& pwm, const std::pair<double, double>& min_max, const unsigned range)
{
    assert(min_max.first <= min_max.second);

    double min = min_max.first;
    const double max = min_max.second;
    if (min == max)
    {
        min = max - 1;        
    }
    min = std::floor(min);

    const unsigned scale = std::floor(range/(max-min));
    ScaledPWM scaled_pwm { /*number_of_sites=*/ pwm.number_of_sites,
                           /*min_before_scaling=*/ int(min),
                           /*scale=*/ scale,
                           /*range=*/ range,
                           /*name=*/ pwm.name };
    scaled_pwm.matrix.reserve(pwm.matrix.size());
    for (auto& row : pwm.matrix)
    {
        std::array<unsigned, AlphabetSize> scaled_row;
        for (size_t alphabet_index=0; alphabet_index < AlphabetSize; ++alphabet_index)
        {
            scaled_row[alphabet_index] = std::round((row[alphabet_index] - min) * scale);  
        }
        scaled_pwm.matrix.push_back(scaled_row);
    }
    return scaled_pwm;
}

// returns score; sequences with invalid characters return 0.
unsigned score(const std::vector<std::array<unsigned, AlphabetSize>>& matrix,
               const std::string& sequence,
               const size_t begin,
               const size_t end)
{
    assert(end >= begin);
    assert((end-begin) <= matrix.size());
    assert(end <= sequence.size());
    /*
    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << matrix[i][j] << ' ';
        } 
        std::cout << '\n';
    }
        std::cout << '\n';
        std::cout << '\n';
        std::cout << '\n';
    */
    unsigned score = 0;
    for (size_t position=begin, row=0; position < end; ++position, ++row)
    {
        const auto column = alphabet_index(sequence[position]);
        if (column >= AlphabetSize)
        {
            // Exceptions can be too slow, since this function will likely be called many times with invalid characters (e.g. 'N').
            // Returning magic number like -1 or max unsigned is error prone.
            // Expected<unsigned> is tempting, but overkill.
            // In practice, we only care about sequences scoring above a threshold, so scoring 
            // 0 for invalid sequences is natural since they won't meet any reasonable threshold.
            return 0;
        }
        score += matrix[row][column];
    }
    return score;
}

unsigned max(const std::vector<std::array<unsigned, AlphabetSize>>& matrix)
{
    unsigned max = 0;
    for (const auto& row : matrix)
    {
        for (unsigned value : row)
        {
            max = std::max(max, value);
        }
    }
    return max;
}

// returns probability distribution values (based on background) indexed by
// all possible integer scores (with the max_score = matrix_max_value * number_of_rows)
std::vector<double>
probability_distribution(const std::vector<std::array<unsigned, AlphabetSize>>& matrix,
                         const std::array<double, AlphabetSize>& background)
{
    const unsigned max_matrix_value = detail::max(matrix);
    const size_t max_score = max_matrix_value * matrix.size();

    // both 0 and max_score are valid scores, so need to add 1 for the vector size
    std::vector<double> prior(max_score + 1, 0);
    std::vector<double> current(prior);

    current[0] = 1; // a score of 0 or better has probability 100%


    //Looping thorugh pwd 
    for (size_t row=0; row < matrix.size(); ++row)
    {
        using std::swap;
        //swap two vectors
        //prior starts at zeros
        swap(prior, current);

        //Max score for row is row number times max_matrix value, which is the max value of the matrix
        //So just normilzaiton
        const size_t max_score_for_row = row*max_matrix_value;
        assert(max_score_for_row <= max_score);

        //fill the current dist with zeros
        std::fill(current.begin(), current.end(), 0);

        //for ACTG
        for (size_t column=0; column < AlphabetSize; ++column)
        {
            //get value at matrix
            const unsigned matrix_score = matrix[row][column];
            assert(matrix_score <= max_score);
            //max score for row
            for (size_t score=0; score <= max_score_for_row; ++score)
            {
                assert(score <= max_score);
                const double prior_probability = prior[score];
                if (prior_probability != 0)
                {
                    assert((score+matrix_score) <= max_score);
                    current[score+matrix_score] += prior_probability * background[column];
                }
            }
        }
    }


    return current;
}

void pdf_to_pvalues(std::vector<double>& p)
{
    if (p.size() <= 1) return;

    for (size_t i=p.size() - 2; ; --i)
    {
        p[i] = std::min(1.0, p[i] + p[i+1]);
        if (i == 0)
        {
            break;
        }
    }
}

std::string bps = "TCGT";
 /*
void create_map_recurse(std::string seq, my_map &matches, std::vector<std::array<unsigned, 4>> matrix, std::vector<double> pvals, unsigned score, unsigned length, int max_score, unsigned cutoff, double m_scale, double m_min_before_scaling) {

    if (score > pvals.size()) {
        return;
    }

    if (seq.length() == length  && pvals[score] < 0.0001) {
        matches[seq][0] = pvals[score];
        matches[seq][1] = double(score)/m_scale + length * m_min_before_scaling;
        return;
    } else if ((length - seq.length()) * max_score + score < cutoff) {
        return;
    } else {
        int letter = 0;
        for (std::string::iterator it = bps.begin(); it != bps.end(); ++it) {
            unsigned tmp_score = score + matrix[seq.length() ][letter];
            create_map_recurse(seq + *it, matches, matrix, pvals, tmp_score, length, max_score, cutoff, m_scale, m_min_before_scaling);
            ++letter;
         }
    }
}
    */

void create_map_recurse_int(uint64_t seq_key, uint64_t bitshifts, unsigned track_length, my_map &matches, std::vector<std::array<unsigned, 4>> matrix, std::vector<double> pvals, unsigned score, unsigned length, int max_score, unsigned cutoff, double m_scale, double m_min_before_scaling) {

    if (score > pvals.size()) {
        return;
    }
    
    if ( track_length == length  && pvals[score] < 0.0001) {
        matches[seq_key][0] = pvals[score];
        matches[seq_key][1] = double(score)/m_scale + length * m_min_before_scaling;
        return;
    } else if ((length - track_length) * max_score + score < cutoff || track_length == length) {
        return;
    } else {
        for (int i = 0; i < 4; ++i) {
            char c = bps[i] + 10;
            seq_key ^= (!!(c & (1 << 2))) << bitshifts;
            seq_key ^= (!!((bps[i] + 10) & (1 << 4))) << (bitshifts - 1);
            create_map_recurse_int(seq_key, bitshifts - 2, track_length + 1, matches, matrix, pvals, score + matrix[track_length][i], length, max_score, cutoff, m_scale, m_min_before_scaling);
         }
    }

}

void create_map(my_map &matches, std::vector <std::array<unsigned, 4>> matrix, std::vector<double> pvals, double m_scale, double m_min_before_scaling) {

    unsigned length = matrix.size();
    unsigned i = 0;
    
    for (i = 0; i < pvals.size(); ++i) {
        if (pvals[i] < 0.0001) {
            break;
        }
    }

    unsigned cutoff = i;
    unsigned max_score = 1000;
    unsigned track_length = 0;
    unsigned bitshifts = length * 2;
//levels starts at the size of the sequence and then goes down
    //create_map_recurse("", matches, matrix, pvals, 0, length, max_score, cutoff, m_scale, m_min_before_scaling);
    create_map_recurse_int(0, bitshifts, track_length,  matches, matrix, pvals, 0, length, max_score, cutoff, m_scale, m_min_before_scaling);

}



void reverse_complement(std::vector<std::array<double, AlphabetSize>>& matrix)
{
    std::reverse(matrix.begin(), matrix.end());
    for (auto& row: matrix)
    {
        std::reverse(row.begin(), row.end());
    }
}

std::array<double, AlphabetSize> adjust_background(std::array<double, AlphabetSize> background, bool average_for_reverse)
{
    if (average_for_reverse)
    {
        // average A and T
        background[0] = (background[0] + background[3])/2.0;
        background[3] = background[0];

        // average C and G
        background[1] = (background[1] + background[2])/2.0;
        background[2] = background[1];
    }

    const double length = background[0] + background[1] + background[2] + background[3];
    if (length != 1.0)
    {
        for (double& frequency : background)
        {
            frequency = frequency/length;
        }
    }

    return background;
}

} }

#endif

/* The MIT License (MIT) 

   Copyright (c) 2015 John DiMatteo (jdimatteo@gmail.com)

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE. 
 */
