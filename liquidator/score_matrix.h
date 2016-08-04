#ifndef LIQUIDATOR_SCORE_MATRIX_H_INCLUDED
#define LIQUIDATOR_SCORE_MATRIX_H_INCLUDED

#include "liquidator_util.h"

#include <array>
#include <cctype>
#include <iostream>
#include <string>
#include <vector>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>
#include <thrust/generate.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/tuple.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#endif

namespace liquidator
{


// motif position weight matrix (pwm) for scoring sequences 
class ScoreMatrix
{
public:

    // See fimo_style_printer.h for example of a ScoreConsumer.
    static constexpr std::array<double, AlphabetSize> default_acgt_background = {{0.281774, 0.222020, 0.228876, 0.267330}};

    // Input format described at http://meme.ebi.edu.au/meme/doc/meme-format.html .
    // Psuedo count logic described at http://meme-suite.org/doc/general-faq.html .
    static std::vector<ScoreMatrix> read(std::istream& meme_style_pwm,
                                         const std::array<double, AlphabetSize>& acgt_background = default_acgt_background,
                                         bool include_reverse_complement = true,
                                         double pseudo_sites = 0.1);

    // Background format described at http://meme.ebi.edu.au/meme/doc/bfile-format.html .
    // Note that only order 0 values are used.
    static std::array<double, AlphabetSize> read_background(std::istream& background);


    ScoreMatrix(const std::string& name,
                const std::array<double, AlphabetSize>& background,
                bool average_background_for_reverse,
                const std::vector<std::array<double, AlphabetSize>>& pwm,
                unsigned number_of_sites,
                bool is_reverse_complement = false,
                double pseudo_sites = 0.1);
    

    // Scores reference a sequence string so are intended to be used only
    // in the scope of a ScoreConsumer operator.
    class Score
    {
        public:
            // writes the matched sequence
            friend std::ostream& operator<<(std::ostream& out, const Score& score);
            
            // returns a copy of the matched sequence
            inline std::string matched_sequence() const;

            // The pvalue, or NAN if sequence was not scorable.
            // Note that NaN < x is false for any double x.
            double pvalue() const { return m_pvalue; }

            // The score, or 0 if sequence was not scorable.
            double score() const { return m_score; }

            bool is_reverse_complement() const { return m_is_reverse_complement; }

            Score(const std::string& sequence, bool is_reverse_complement, size_t begin, size_t end, double pvalue, double score);
            struct score_data {
                const bool m_is_reverse_complement;
                const size_t m_begin;
                const size_t m_end;
                const double m_pvalue;
                const double m_score;
            };


        private:
            const std::string& m_sequence;
            const bool m_is_reverse_complement;
            const size_t m_begin;
            const size_t m_end;
            const double m_pvalue;
            const double m_score;
    };
   
    #ifdef __CUDACC__
    /*scaled score, p-value, start loc, end loc*/ 
    typedef thrust::tuple<double, double, size_t, size_t> score_values;

    struct FunctorScore {
        thrust::device_ptr <double> p_value;
        thrust::device_ptr <unsigned> matrix;
        thrust::device_ptr <char> sequence;
        size_t sequence_length;
        size_t motif_size;
        size_t pvalue_size;
        double const_unscaled;
        double m_scale;

        FunctorScore() {};

        void set_matrix(std::vector<std::array<unsigned, 4>> test) {
            matrix = thrust::device_malloc<unsigned>(test.size() * 4);
            for (int r = 0; r < test.size(); ++r) {
                for (int c = 0; c < 4; ++c) {
                    matrix[r * 4 + c] = test[r][c];
                }
            }   

        }

        void set_pvalue(std::vector<double> pval) {
            p_value = thrust::device_malloc<double> (pval.size());
            for (int i = 0; i < pval.size(); ++i) {
                p_value[i] = pval[i];
            }

        }

        __device__
        unsigned pos(char c) {

            if (c == 'A') {
                return 0;
            } else if (c == 'T') {
                return 3;
            } else if (c == 'C') {
                return 1;
            } else if (c == 'G') {
                return 2;
            } else {
                return 99;
            }

        };

        __device__
        score_values operator() (int i) {
            unsigned score = 0;
            int position = 0;
            score_values sv;
            /*set large pvalue*/
            sv.get<1>() = 1;

            for (int j = i; j < i + motif_size; j++) {
                if ((position = pos(sequence[j])) != 99) {
                    score += matrix[position + ((j-i) * 4)];
                } else {
                    return sv;
                }
            }
            
            sv.get<1>() = p_value[score];
            sv.get<0>() = double(score)/m_scale + const_unscaled;
            sv.get<2>() = i;
            sv.get<3>() = i + motif_size;
            return sv;
        }
            
    };

    /*Binary operator that returns smallest p-value*/
    struct p_min {
        __device__
        score_values operator() (const score_values &tmp1, const score_values &tmp2) {
            return tmp1.get<1>() < tmp2.get<1>() ? tmp1 : tmp2;
        }
    };

    struct FunctorScore customFunctor;
    #endif

    template <typename ScoreConsumer>
    void score(const std::string& sequence, ScoreConsumer& consumer) 
    {
        #ifdef __CUDACC__
            thrust::device_vector<char> seq(sequence.length());
            for (int i = 0; i < sequence.length(); ++i) {
                seq[i] = sequence[i];
            } 
            customFunctor.sequence = seq.data(); 
        
            //Set large pvalue
            score_values tmpSv;
            tmpSv.get<1>() = 1; 
        
            score_values tmpScore  = thrust::transform_reduce(thrust::device, thrust::make_counting_iterator((int)0), thrust::make_counting_iterator((int)(sequence.length() - m_matrix.size())), customFunctor, tmpSv, p_min());
            consumer(m_name, tmpScore.get<2>(), tmpScore.get<3>(), Score(sequence, m_is_reverse_complement, tmpScore.get<2>(), tmpScore.get<3>(), tmpScore.get<1>(), tmpScore.get<0>()));
         
            /* thrust::device_vector<score_values> sv(sequence.length());
            thrust::transform(thrust::device, thrust::make_counting_iterator((int)0), thrust::make_counting_iterator((int)(sequence.length() - m_matrix.size())), sv.begin(), customFunctor);
        */
        #else
            for (size_t start = 1, stop = m_matrix.size(); stop <= sequence.size(); ++start, ++stop)
            {
                const Score score = score_sequence(sequence, start-1, stop);
                consumer(m_name, start, stop, score);
            }
        #endif
    }

    std::string name() { return m_name; }
    size_t length() { return m_matrix.size(); }

    // Matrix value for the sequence position (row) and base letter (column).
    // Value is a log likelihood ratio, adjusted with a psuedo count and scaled.
    // Base should be ACGT/acgt -- else throws exception.
    // position is 0 based and must be < length() -- else undefined behavior.
    int value(size_t position, char base)
    {
        const auto column = alphabet_index(base);
        if ( column >= AlphabetSize ) throw std::runtime_error("Invalid base " + std::string(1, base));
        return m_matrix[position][column];
    }

private:
    const std::string m_name;
    const bool m_is_reverse_complement;
    std::vector<std::array<unsigned, AlphabetSize>> m_matrix;
    double m_scale;
    double m_min_before_scaling;
    std::vector<double> m_pvalues;

    Score score_sequence(const std::string& sequence, size_t begin, size_t end);
};

inline std::ostream& operator<<(std::ostream& out, const ScoreMatrix::Score& score)
{
    if (score.m_is_reverse_complement)
    {
        if (score.m_end > score.m_begin)
        {
            for (size_t i=score.m_end-1; ; --i)
            {
                out << char(std::toupper(complement(score.m_sequence[i])));
                if ( i == score.m_begin )
                {
                    break;
                }
            }
        }
    }
    else
    {
        for (size_t i=score.m_begin; i < score.m_end; ++i)
        {
            out << char(std::toupper(score.m_sequence[i]));
        }
    }
    return out;
}

}
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
#endif
