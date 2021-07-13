---
layout: post
comments: true
title:  "Neural Network C++"
excerpt: "Implementasi supervised learning pada neural network sederhana dengan C++"
date:   2017-07-04 22:00:00
---

Mungkin kamu akan banyak menemukan tulisan seperti ini di internet, akan tetapi saya akan mengulangnya untuk tujuan belajar yang lebih baik. Kode C++ ini akan menunjukkan tugas supervised learning menggunakan neural network yang sangat sederhana.

**Komponen inti dari kode, algoritma pembelajaran, hanya 10 baris:**

```c++
int main(int argc, const char * argv[]) {    
    for (unsigned i = 0; i != 50; ++i) {        
        vector<float> pred = sigmoid(dot(X, W, 4, 4, 1 ) ); // <--- Baris 3
        vector<float> pred_error = y - pred;  // <--- Baris 4     
        vector<float> pred_delta = pred_error * sigmoid_d(pred); // <--- Baris 5      
        vector<float> W_delta = dot(transpose( &X[0], 4, 4 ), pred_delta, 4, 4, 1); // Baris 6     
        W = W + W_delta; // <--- Baris 7
    };
    return 0;
}
```

Loop di atas berjalan selama 50 iterasi (epochs) dan cocok dengan vektor atribut `X` ke vektor kelas `y` melalui vektor bobot `W`. Saya akan menggunakan 4 catatan dari [dataset bunga Iris](https://en.wikipedia.org/wiki/Iris_flower_data_set). Atribut `X` adalah panjang sepal, lebar sepal, panjang kelopak, dan lebar kelopak. Dalam contoh saya, saya memiliki 2 (Iris Setosa **0** dan Iris Virginica **1**) dari 3 kelas yang dapat Anda temukan di dataset asli. Prediksi disimpan dalam vektor `pred`. Arsitektur neural network. Nilai vektor `W` dan perubahan `pred` selama pelatihan jaringan, sedangkan vektor `X` dan `y` tidak boleh diubah:

```c++
       X            W     pred    y
5.1 3.5 1.4 0.2    0.5    0.00    0
4.9 3.0 1.4 0.2    0.5    0.00    0
6.2 3.4 5.4 2.3    0.5    0.99    1
5.9 3.0 5.1 1.8    0.5    0.99    1
```

Ukuran matriks `X` adalah ukuran batch dengan jumlah atribut.

**Baris 3**. Buat prediksi:

```c++
vector pred = sigmoid(dot(X, W, 4, 4, 1 ) );
```

Untuk menghitung prediksi, pertama-tama, kita perlu mengalikan 4 x 4 matriks `X` dengan 4 x 1 matriks `W`. Kemudian, kita perlu menerapkan [fungsi aktivasi](https://en.wikipedia.org/wiki/Activation_function); dalam hal ini, kita akan menggunakan [fungsi sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function).

Subrutin untuk [perkalian matriks](https://en.wikipedia.org/wiki/Matrix_multiplication):

```c++
vector <float> dot (const vector <float>& m1, const vector <float>& m2, 
                    const int m1_rows, const int m1_columns, const int m2_columns) {
    
    /*  Mengembalikan produk dari dua matriks: m1 x m2.
        Inputs:
            m1: vektor, matriks kiri ukuran m1_rows x m1_columns
            m2: vektor, matriks kanan ukuran m1_columns x m2_columns
                (jumlah baris dalam matriks kanan harus sama dengan
                jumlah kolom di sebelah kiri)
            m1_rows: int, jumlah baris dalam matriks kiri m1
            m1_columns: int, jumlah kolom dalam matriks kiri m1
            m2_columns: int, jumlah kolom dalam m2 matriks kanan
        Output: vektor, m1 * m2, produk dari dua vektor m1 dan m2,
                matriks ukuran m1_rows x m2_columns
    */
    
    vector <float> output (m1_rows*m2_columns);
    
    for( int row = 0; row != m1_rows; ++row ) {
        for( int col = 0; col != m2_columns; ++col ) {
            output[ row * m2_columns + col ] = 0.f;
            for( int k = 0; k != m1_columns; ++k ) {
                output[ row * m2_columns + col ] += m1[ row * m1_columns + k ] * m2[ k * m2_columns + col ];
            }
        }
    }
    
    return output;
}
```

Subrutin untuk fungsi sigmoid:

```c++
vector <float> sigmoid (const vector <float>& m1) {
    
    /*  Mengembalikan nilai fungsi sigmoid f (x) = 1 / (1 + e ^ -x).
        Input: m1, sebuah vektor.
        Output: 1 / (1 + e ^ -x) untuk setiap elemen dari matriks input m1.
    */
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> output (VECTOR_SIZE);
    
    
    for( unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        output[ i ] = 1 / (1 + exp(-m1[ i ]));
    }
    
    return output;
}
```

Fungsi sigmoid (merah) dan turunan pertamanya (grafik biru):

{:refdef: style="text-align: center;"}
<img src="/assets/cppnn/nncpp-01.png" height="300">
{:refdef}

**Baris 4**. Hitung `pred_error`, itu hanya perbedaan antara prediksi dan kebenaran:

```c++
vector<float> pred_error = y - pred;
```

Untuk mengurangi satu vektor dari yang lain, kita perlu membebani operator `-`:

```c++
vector <float> operator-(const vector <float>& m1, const vector <float>& m2){
    
    /*  Mengembalikan perbedaan antara dua vektor.
        Inputs:
            m1: vektor
            m2: vektor
        Output: vektor, m1 - m2, m1 - m2, perbedaan antara dua vektor m1 dan m2.
    */
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> difference (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        difference[i] = m1[i] - m2[i];
    };
    
    return difference;
}
```

**Baris 5**. Tentukan vektor delta `pred_delta`:

```c++
vector<float> pred_delta = pred_error * sigmoid_d(pred);
```

Untuk melakukan perkalian dua vektor secara elemetwise, kita perlu membebani operator `*` secara berlebihan:

```c++
vector <float> operator*(const vector <float>& m1, const vector <float>& m2){
    
    /*  Mengembalikan produk dari dua vektor (perkalian elemen).
        Inputs:
            m1: vektor
            m2: vektor
        Output: vektor, m1 * m2, produk dari dua vektor m1 dan m2
    */
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        product[i] = m1[i] * m2[i];
    };
    
    return product;
}
```

Subrutin untuk turunan dari fungsi sigmoid `d_sigmoid` :

Pada dasarnya, kami menggunakan turunan pertama untuk menemukan kemiringan garis singgung dengan grafik fungsi sigmoid. Pada `x = 0` kemiringan sama dengan **0,25**. Semakin jauh prediksi dari **0**, semakin dekat kemiringan ke **0**: pada `x = ± 10` kemiringan sama dengan **0,000045**. Oleh karena itu, delta akan menjadi kecil jika kesalahannya kecil atau jaringan sangat yakin tentang prediksi (yakni `abs(x)` lebih besar dari **4**).

```c++
vector <float> sigmoid_d (const vector <float>& m1) {
    
    /*  Mengembalikan nilai turunan fungsi sigmoid f '(x) = f (x) (1 - f (x)), di mana f (x) adalah fungsi sigmoid.
        Input: m1, a vektor.
        Output: x(1 - x) untuk setiap elemen dari matriks input m1.
    */
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> output (VECTOR_SIZE);
    
    
    for( unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        output[ i ] = m1[ i ] * (1 - m1[ i ]);
    }
    
    return output;
}
```

**Baris 6**. Hitung `W_delta` :

Baris ini menghitung pembaruan berat. Untuk melakukan itu, kita perlu melakukan perkalian matriks dari matriks `X` yang ditransfosisikan dengan matriks `pred_delta`.

```c++
vector W_delta = dot(transpose( &X[0], 4, 4 ), pred_delta, 4, 4, 1);
```

Subrutin yang [mentransposisi](https://en.wikipedia.org/wiki/Transpose) matriks:

```c++
vector <float> transpose (float *m, const int C, const int R) {
    
    /*  Mengembalikan matriks transpose dari matriks input.
        Inputs:
            m: vektor, input matrix
            C: int, jumlah kolom dalam matriks input
            R: int, jumlah baris dalam matriks input
        Output: vektor, transpos matriks mT dari matriks input m
    */
    
    vector <float> mT (C*R);
    
    for(int n = 0; n!=C*R; n++) {
        int i = n/C;
        int j = n%C;
        mT[n] = m[R*j + i];
    }
    
    return mT;
}
```

**Baris 7**. Perbarui bobot `W`:

```c++
W = W + W_delta;
```

Untuk melakukan operasi penambahan matriks, kita perlu membebani operator `+` :

```c++
vector <float> operator+(const vector <float>& m1, const vector <float>& m2){
    
    /*  Mengembalikan jumlah elementwise dari dua vektor.
        Inputs: 
            m1: sebuah vektor
            m2: sebuah vektor
        Output: sebuah vektor, jumlah vektor m1 dan m2.
    */
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> sum (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        sum[i] = m1[i] + m2[i];
    };
    
    return sum;
}
```

Kamu bisa melihat kode lengkapnnya di GitHub Gist berikut:

<script src="https://gist.github.com/prasetyoandi/595c67df6893b082e95aebe48c24ba2d.js"></script>

Output:

```c++
0.0511965 
0.0696981 
0.931842 
0.899579 

Program ended with exit code: 0
```
