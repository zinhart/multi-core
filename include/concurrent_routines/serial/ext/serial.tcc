#include <iostream>
#include <cmath>
namespace zinhart
{
  namespace serial
  {
	// taken from wikipedia https://en.wikipedia.org/wiki/Kahan_summation_algorithm	
	template <class precision_type>
	  HOST precision_type kahan_sum(precision_type * data, const std::uint32_t & data_size)
	  {
		precision_type * temp = data;
		precision_type sum{*temp};
		// a running compensation for lost lower-order bits
		precision_type compensation{0.0}; 		
		++temp;
		for(std::uint32_t i  = 1; i < data_size; ++i)
		{
		  precision_type y{*temp - compensation};
		  // lower order bits are lost here with this addition
		  precision_type t{sum + y}; 
		  // (t - sum) cancels the higher order part of y and subtracting y recorvers the low part of y
		  compensation = (t - sum) - y;		  
		  sum = t;
		  ++temp;
		}
		return sum;
	  }

	// taken from wikipedia, this is an improvement on the the algo above https://en.wikipedia.org/wiki/Kahan_summation_algorithm
	template <class precision_type>
	  HOST precision_type neumaier_sum(precision_type * data, const std::uint32_t & data_size)
	  {
		precision_type * temp = data;
		precision_type sum{*temp};
		// a running compensation for lost lower-order bits
		precision_type compensation{0.0}; 		
		++temp;
		for(std::uint32_t i = 1; i < data_size; ++i)
		{
		  precision_type t{sum + data[i]};
		  if(std::abs(sum) >= std::abs(*temp))
			// if the sum is bigger lower order digitis of in[i] are lost
			compensation += (sum - t) + *temp;
		  else
			// if the sum is smaller lower order digits of sum are lost
			compensation += (*temp - t) + sum;
		  sum = t;
		  ++temp;
		}
		// Correction is applied once
		return sum + compensation;
	  }

	template <class precision_type, class binary_predicate>
	  HOST precision_type kahan_sum(precision_type * vec_1, precision_type * vec_2, const std::uint32_t & data_size, binary_predicate bp)
	  {
		precision_type * v1{vec_1}, * v2{vec_2};
		precision_type post_bp{bp(*v1, *v2)};
		precision_type sum{ post_bp };
		// a running compensation for lost lower-order bits
		precision_type compensation{0.0}; 		
		for(std::uint32_t i  = 1; i < data_size; ++i)
		{
		  ++v1;
		  ++v2;
		  post_bp = bp(*v1, *v2);
		  precision_type y{ post_bp - compensation };
		  // lower order bits are lost here with this addition
		  precision_type t{sum + y}; 
		  // (t - sum) cancels the higher order part of y and subtracting y recorvers the low part of y
		  compensation = (t - sum) - y;		  
		  sum = t;
		}
		return sum;
	  }
	template <class precision_type, class binary_predicate>
	  HOST precision_type neumaier_sum(precision_type * vec_1, precision_type * vec_2, const std::uint32_t & data_size, binary_predicate bp)
	  {
		precision_type * v1{vec_1}, * v2{vec_2};
		precision_type post_bp{bp(*v1, *v2)};
		precision_type sum{ post_bp };
		// a running compensation for lost lower-order bits
		precision_type compensation{0.0}; 		
		for(std::uint32_t i = 1; i < data_size; ++i)
		{
		  ++v1;
		  ++v2;
		  post_bp = bp(*v1, *v2);
		  precision_type t{sum + post_bp};
		  if(std::abs(sum) >= std::abs(post_bp))
			// if the sum is bigger lower order digits of in[i] are lost
			compensation += (sum - t) + post_bp;
		  else
			// if the sum is smaller lower order digits of sum are lost
			compensation += ( post_bp - t) + sum;
		  sum = t;
		}
		// Correction is applied once
		return sum + compensation;
	  }
	template<class precision_type>
	  HOST void serial_matrix_product(const precision_type * A, const precision_type * B, precision_type * C, const std::uint32_t M, const std::uint32_t N, const std::uint32_t K)
	  {
		for(std::uint32_t a = 0; a < M; ++a)
		  for(std::uint32_t b = 0; b < N; ++b)
  			for(std::uint32_t c = 0; c < K; ++c)
			  C[idx2r(a, b, N)] += A[idx2r(a, c, K)] * B[idx2r(c, b, N)];
	  }

	template<class precision_type>
	  HOST void cache_aware_serial_matrix_product(const precision_type * A, const precision_type * B, precision_type * C, const std::uint32_t M, const std::uint32_t N, const std::uint32_t K)
	  {
		for(std::uint32_t a = 0; a < M; ++a)
		  for(std::uint32_t c = 0; c < K; ++c)
  			for(std::uint32_t b = 0; b < N; ++b)
			  C[idx2r(a, b, N)] += A[idx2r(a, c, K)] * B[idx2r(c, b, N)];
  	  }
	template<class precision_type>
	  HOST void print_matrix_row_major(precision_type * mat, std::uint32_t mat_rows, std::uint32_t mat_cols, std::string s)
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

