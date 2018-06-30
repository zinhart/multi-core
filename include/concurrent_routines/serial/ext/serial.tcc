#include <iostream>
#include <cmath>
namespace zinhart
{
  namespace serial
  {
	// taken from wikipedia https://en.wikipedia.org/wiki/Kahan_summation_algorithm	
	template <class Precision_Type>
	  HOST Precision_Type kahan_sum(Precision_Type * data, const std::uint32_t & data_size)
	  {
		Precision_Type sum{data[0]};
		// a running compensation for lost lower-order bits
		Precision_Type compensation{0.0}; 		
		for(std::uint32_t i  = 1; i < data_size; ++i)
		{
		  Precision_Type y{data[i] - compensation};
		  // lower order bits are lost here with this addition
		  Precision_Type t{sum + y}; 
		  // (t - sum) cancels the higher order part of y and subtracting y recorvers the low part of y
		  compensation = (t - sum) - y;		  
		  sum = t;
		}
		return sum;
	  }

	// taken from wikipedia, this is an improvement on the the algo above https://en.wikipedia.org/wiki/Kahan_summation_algorithm
	template <class Precision_Type>
	  HOST Precision_Type neumaier_sum(Precision_Type * data, const std::uint32_t & data_size)
	  {
		Precision_Type sum{data[0]};
		// a running compensation for lost lower-order bits
		Precision_Type compensation{0.0}; 		
		for(std::uint32_t i = 1; i < data_size; ++i)
		{
		  Precision_Type t{sum + data[i]};
		  if(std::abs(sum) >= std::abs(data[i]))
			// if the sum is bigger lower order digitis of in[i] are lost
			compensation += (sum - t) + data[i];
		  else
			// if the sum is smaller lower order digits of sum are lost
			compensation += (data[i] - t) + sum;
		  sum = t;
		}
		// Correction is applied once
		return sum + compensation;
	  }
	template<class Precision_Type>
	  HOST void serial_matrix_product(const Precision_Type * A, const Precision_Type * B, Precision_Type * C, const std::uint32_t M, const std::uint32_t N, const std::uint32_t K)
	  {
		for(std::uint32_t a = 0; a < M; ++a)
		  for(std::uint32_t b = 0; b < K; ++b)
			for(std::uint32_t c = 0; c < N; ++c)
			  C[idx2r(a, b, K)] += A[idx2r(a, c, N)] * B[idx2r(c, b, K)];
	  }

	template<class Precision_Type>
	  HOST void cache_aware_serial_matrix_product(const Precision_Type * A, const Precision_Type * B, Precision_Type * C, const std::uint32_t M, const std::uint32_t N, const std::uint32_t K)
	  {
		for(std::uint32_t a = 0; a < M; ++a)
		  for(std::uint32_t c = 0; c < N; ++c)
			for(std::uint32_t b = 0; b < K; ++b)
			  C[idx2r(a, c, K)] += A[idx2r(a, b, N)] * B[idx2r(c, b, K)];
	  }
	template<class Precision_Type>
	  HOST void print_matrix_row_major(Precision_Type * mat, std::uint32_t mat_rows, std::uint32_t mat_cols, std::string s)
	  {
		 std::cout<<s<<"\n";
		 for(std::uint32_t i = 0; i < mat_rows; ++i)  
		 {
		   for(std::uint32_t j = 0; j < mat_cols; ++j)
		   {
			 std::cout<<mat[idx2r(i,j,mat_cols)]<<" ";
		   }
		   std::cout<<"\n";
		 }
	  }
  }// END NAMESPACE SERIAL
}// END NAMESPACE ZINHART

