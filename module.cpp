#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>

// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ) {
    return tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
    tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    
    // -------- YOUR CODE HERE  -------- //
    for (int b = 0; b < B; b++){
        for (int h = 0; h < H; h++){

            for (int i = 0; i < N; i++){
                for (int j = 0; j < N; j++){
                    float temp = 0.0;
                    for(int k = 0; k < d; k++){
                        temp += fourDimRead(Q, b, h, i, k, H, N, d) * fourDimRead(K, b, h, j, k, H, N, d);
                    }
                    twoDimWrite(QK_t, i, j, N, temp);
                }
            }

            for(int i = 0;  i < N; i++){
                float exp_sum = 0.0;
                for(int j = 0; j < N; j++){
                    float element =  twoDimRead(QK_t, i, j, N);
                    element = std::exp(element);
                    exp_sum += element;
                    twoDimWrite(QK_t, i, j, N, element);
                }
                for(int j = 0; j < N; j++){
                    float element =  twoDimRead(QK_t, i, j, N);
                    element = element / exp_sum;
                    twoDimWrite(QK_t, i, j, N, element);
                }
            }

            for (int i = 0; i < N; i++){
                for (int j = 0; j < d; j++){
                    float temp = 0.0;
                    for (int k = 0; k < N; k++){
                        temp += twoDimRead(QK_t, i, k, N) * fourDimRead(V, b, h, k, j, H, N, d);
                    }
                    fourDimWrite(O, b, h, i, j, H, N, d, temp);
                }
            }
        }
    }
   
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //
    int tile_size = 16;
    int tile_num_N = (N + tile_size - 1) / tile_size;
    int tile_num_d = (d + tile_size - 1) / tile_size; 
    for (int b = 0; b < B; b++){
        for (int h = 0; h < H; h++){
                       
            for (int i_tile = 0; i_tile < tile_num_N; i_tile++){
                for (int j_tile = 0; j_tile < tile_num_N; j_tile++){
                    int i_begin = i_tile * tile_size;
                    int i_end = i_begin + std::min(tile_size, N - i_tile * tile_size);
                    int j_begin = j_tile * tile_size;
                    int j_end = j_begin + std::min(tile_size, N - j_tile * tile_size);
                    
                    for(int k_tile = 0; k_tile < tile_num_d; k_tile++){
                        int k_begin = k_tile * tile_size;
                        int k_end = k_begin + std::min(tile_size, d - k_tile * tile_size);
                        for(int i = i_begin; i < i_end; i++){
                            for(int j = j_begin; j < j_end; j++){
                                float temp = twoDimRead(QK_t, i, j, N);
                                for(int k = k_begin; k < k_end; k++){
                                    temp += fourDimRead(Q, b, h, i, k, H, N, d) * fourDimRead(K, b, h, j, k, H, N, d);
                                } 
                                twoDimWrite(QK_t, i, j, N, temp);
                            }
                        }
                    }
                }
            }

            for(int i = 0;  i < N; i++){
                float exp_sum = 0.0;
                for(int j = 0; j < N; j++){
                    float element =  twoDimRead(QK_t, i, j, N);
                    element = std::exp(element);
                    exp_sum += element;
                    twoDimWrite(QK_t, i, j, N, element);
                }
                for(int j = 0; j < N; j++){
                    float element =  twoDimRead(QK_t, i, j, N);
                    element = element / exp_sum;
                    twoDimWrite(QK_t, i, j, N, element);
                }
            }

            for (int i_tile = 0; i_tile < tile_num_N; i_tile++){
                for (int j_tile = 0; j_tile < tile_num_d; j_tile++){
                    int i_begin = i_tile * tile_size;
                    int i_end = i_begin + std::min(tile_size, N - i_tile * tile_size);
                    int j_begin = j_tile * tile_size;
                    int j_end = j_begin + std::min(tile_size, d - j_tile * tile_size);
                    
                    for(int k_tile = 0; k_tile < tile_num_N; k_tile++){
                        int k_begin = k_tile * tile_size;
                        int k_end = k_begin + std::min(tile_size, N - k_tile * tile_size);
                        for(int i = i_begin; i < i_end; i++){
                            for(int j = j_begin; j < j_end; j++){
                                float temp = fourDimRead(O, b, h, i, j, H, N, d);
                                for(int k = k_begin; k < k_end; k++){
                                    temp += twoDimRead(QK_t, i, k, N) * fourDimRead(V, b, h, k, j, H, N, d);
                                } 
                                fourDimWrite(O, b, h, i, j, H, N, d, temp);
                            }
                        }
                    }
                }
            }

            
        }
    }
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
    // We give you a template of the first three loops for your convenience
    //loop over batch
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++){

        //loop over heads
        for (int h = 0; h < H; h++){
            for (int i = 0; i < N ; i++){
                
		// YRow is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
                std::vector<float> ORow = formatTensor(ORowTensor);
		//YOUR CODE HERE
                for (int j = 0; j < N; j++) {
                    float temp = 0.0;
                    for (int k = 0; k < d; k++) {
                        temp += fourDimRead(Q, b, h, i, k, H, N, d) * fourDimRead(K, b, h, j, k, H, N, d);
                    }
                    ORow[j] = temp;
                }

                float exp_sum = 0.0;
                for(int j = 0; j < N; j++){
                    ORow[j] = std::exp(ORow[j]);
                    exp_sum += ORow[j];
                }
                for(int j = 0; j < N; j++){
                    ORow[j] /= exp_sum;
                }

                for (int j = 0; j < d; j++){
                    float temp = 0.0;
                    for (int k = 0; k < N; k++){
                        temp += ORow[k] * fourDimRead(V, b, h, k, j, H, N, d);
                    }
                    fourDimWrite(O, b, h, i, j, H, N, d, temp);
                }
                
            }
	}
    }
	    
	
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

            


    // -------- YOUR CODE HERE  -------- //
    int Tr = (N + Br - 1) / Br;
    int Tc = (N + Bc - 1) / Bc;
    for(int b = 0; b < B; b++){
        for(int h = 0; h < H; h++){

            std::vector<float> Sij = formatTensor(SijTensor);
            std::vector<float> Pij = formatTensor(PijTensor);
            std::vector<float> Kj = formatTensor(KjTensor);
            std::vector<float> Vj = formatTensor(VjTensor);
            std::vector<float> Qi = formatTensor(QiTensor);
            std::vector<float> Oi = formatTensor(OiTensor);
            std::vector<float> l = formatTensor(LTensor);
            std::vector<float> PV = formatTensor(PVTensor);
            std::vector<float> li = formatTensor(LiTensor);
            std::vector<float> lij = formatTensor(LijTensor);
            std::vector<float> lnew = formatTensor(LnewTensor);
            for(int j = 0; j < Tc; j++){
                //load Ki, Vj into local memory blocks
                int mc = std::min(Bc, N - j * Bc);
                int r_begin = j * Bc;
                int r_end = r_begin + mc;
                for(int r = r_begin; r < r_end; r++){
                    int r_index = r - r_begin;
                    for(int c = 0; c < d; c++){
                        float k_element = fourDimRead(K, b, h, r, c, H, N, d);
                        float v_element = fourDimRead(V, b, h, r, c, H, N, d);
                        twoDimWrite(Kj, r_index, c, d, k_element);
                        twoDimWrite(Vj, r_index, c, d, v_element);
                    }
                } 

                for(int i = 0; i < Tr; i++){
                    //load Qi, Oi, li into local memory block
                    int mr = std::min(Br, N - i * Br);
                    int r_begin = i * Br;
                    int r_end = r_begin + mr;
                    for(int r = r_begin; r < r_end; r++){
                        int r_index = r - r_begin;
                        for(int c = 0; c < d; c++){
                            float Q_element = fourDimRead(Q, b, h, r, c, H, N, d);
                            float O_element = fourDimRead(O, b, h, r, c, H, N, d);
                            twoDimWrite(Qi, r_index, c, d, Q_element);
                            twoDimWrite(Oi, r_index, c, d, O_element);                           
                        }               
                        li[r_index] = l[r];
                    }                  

                    //Sij = Qi * Kj ^ T
                    for(int r = 0; r < mr; r++){
                        for(int c = 0; c < mc; c++){
                            float temp = 0.0;
                            for(int k = 0; k < d; k++){
                                temp += twoDimRead(Qi, r, k, d) * twoDimRead(Kj, c, k, d);
                            }
                            twoDimWrite(Sij, r, c, Bc, temp);
                        }
                    }

                    //Pij = exp(Sij)
                    for(int r = 0; r < mr; r++){
                        for(int c = 0; c < mc; c++){
                            float temp =  std::exp(twoDimRead(Sij, r, c, Bc));
                            twoDimWrite(Pij, r, c, Bc, temp);
                        }
                    }

                    //lij = rowsum(Pij)
                    for(int r = 0; r < mr; r++){
                        float temp = 0.0;
                        for(int c = 0; c < mc; c++){
                            temp += twoDimRead(Pij, r, c, Bc);
                        }
                        lij[r] = temp;
                    }

                    //lnew = li + lij
                    for(int r = 0; r < mr; r++){
                        lnew[r] = li[r] + lij[r];
                    }

                    //Oi = (l * Oi + Pij * Vj) / lnew
                    for(int r = 0; r < mr; r++){
                        for(int c = 0; c < d; c++){
                            float temp = 0.0;   
                            temp += li[r] * twoDimRead(Oi, r, c, d);               
                            for(int k = 0; k < mc; k++){
                                temp += twoDimRead(Pij, r, k, Bc) * twoDimRead(Vj, k, c, d);
                            }
                            temp /= lnew[r];
                            twoDimWrite(Oi, r, c, d, temp);
                        }
                    }

                    //write back Oi and lnew to O and l in main memory
                    for(int r = 0;  r < mr; r++){
                        int r_index = i * Br + r;
                        l[r_index] = lnew[r];
                        for(int c = 0; c < d; c++){
                            float Oi_element = twoDimRead(Oi, r, c, d);
                            fourDimWrite(O, b, h, r_index, c, H, N, d, Oi_element);
                        }
                    }
                }
            }
        }
    }


    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
