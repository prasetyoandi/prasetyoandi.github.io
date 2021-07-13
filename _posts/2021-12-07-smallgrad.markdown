---
layout: post
comments: true
title:  "Smallgrad"
excerpt: "Implementasi reverse-mode diferensiasi otomastis(autograd/backpropagation) 'mini' dengan Python"
date:   2021-06-11 22:00:00
mathjax: false
---

Smallgrad adalah Implementasi reverse-mode diferensiasi otomastis(autograd/backpropagation) "kecil" dengan Python[^1]. Smallgrad diimplementasikan dalam satu kelas python kecil(~100 loc), tanpa menggunakan modul eksternal(pure python). Entri logika diferensiasi otomatis ada dalam kelas `scalar` pada smallgrad. Scalar ini membungkus float / integer dan mengesampingkan [arithmetic magic method](https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types)-nya untuk:
- Menyatukan grafik komputasi yang ditentukan untuk melakukan operasi aritmatika pada `scalar` saat dijalankan
- Men-kode-kan fungsi turunan dari operasi aritmatika
- Melacak `∂self/∂parent` di antara node yang berdekatan
- Menghitung `∂output/∂self` dengan aturan sesuai kebutuhan(saat `.backward()` dipanggil)

Ini disebut [reverse-mode automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation). Ini sangat berguna saat kamu punya beberapa output dan banyak input, karena menghitung semua turunan dari satu output dalam satu putaran.

## Struktur

Berikut adalah seluruh logika autograd dalam satu kelas numerik:

```python

class Scalar:
    def __init__(self, value, parents=[], parent_op=None):
        self.value = value
        self.parents = parents
        self.parent_op = parent_op
        self.grad = 0           # untuk menyimpan nilai ∂output/∂self
        self.grad_wrt = dict()  # untuk menyimpan semua nilai ∂self/∂parent
                                # (hanya populated jika self mempunyai parents, yaitu self berasal dari operasi aritmatika)

    def __repr__(self):
        return f'Scalar(value={self.value:.2f}, grad={self.grad:.2f})' 
    
    # dipanggil oleh: self + other
    def __add__(self, other):
        if not isinstance(other, Scalar): other = Scalar(other)
        output = Scalar(self.value + other.value, [self, other], '+')
        
        output.grad_wrt[self] = 1
        output.grad_wrt[other] = 1
        
        return output
    
    # dipanggil oleh: other + self
    def __radd__(self, other):
        return self.__add__(other)
    
    # dipanggl oleh: self - other
    def __sub__(self, other):
        if not isinstance(other, Scalar): other = Scalar(other)
        output = Scalar(self.value - other.value, [self, other], '-')
            
        output.grad_wrt[self] = 1
        output.grad_wrt[other] = -1 
        
        return output
    
    # dipanggil oleh: other - self
    def __rsub__(self, other):
        if not isinstance(other, Scalar): other = Scalar(other)
        output = Scalar(other.value - self.value, [self, other], '-')
        
        output.grad_wrt[self] = -1
        output.grad_wrt[other] = 1
            
        return output
    
    # dipanggil oleh: self * other
    def __mul__(self, other):
        if not isinstance(other, Scalar): other = Scalar(other)
        output = Scalar(self.value * other.value, [self, other], '*')
        
        output.grad_wrt[self] = other.value
        output.grad_wrt[other] = self.value

        return output
    
    # dipanggil oleh: other * self
    def __rmul__(self, other):
        return self.__mul__(other)

    # dipanggil oleh: self / other
    def __truediv__(self, other):
        if not isinstance(other, Scalar): other = Scalar(other)
        output = Scalar(self.value / other.value, [self, other], '/')
        
        output.grad_wrt[self] = 1 / other.value
        output.grad_wrt[other] = -self.value / other.value**2
        
        return output
    
    # dipanggil oleh: other / self
    def __rtruediv__(self, other):
        if not isinstance(other, Scalar): other = Scalar(other)
        output = Scalar(other.value / self.value, [self, other], '/')
        
        output.grad_wrt[self] = -other.value / self.value**2
        output.grad_wrt[other] = 1 / self.value
            
        return output
  
    # dipanggil oleh: self**other
    def __pow__(self, other):
        assert isinstance(other, (int, float)), '''smallgrad tidak support scalar pada eksponen'''
        output = Scalar(self.value ** other, [self], f'^{other}')
        
        output.grad_wrt[self] = other * self.value**(other - 1)

        return output
    
    # dipanggil oleh: -self
    def __neg__(self):
        return self.__mul__(-1)
    
    # dipanggil oleh: self.relu()
    def relu(self):
        output = Scalar(max(self.value, 0), [self], 'relu')
        
        output.grad_wrt[self] = int(self.value > 0)
        
        return output
    
    def backward(self):
        '''hitung ∂self/∂node, yaitu ∂output/∂node, untuk setiap node pada grafik dependensi self.'''

        # Note: untuk melakukan reverse-mode autodiff dengan benar, kita perlu untuk melintasi DAG 
        # tepatnya dalam urutan dari output kepada input, mencapai setiap node sekali, dan melengkapi
        # komputasi gradient(lihat _compute_grad_of_parents()) pada langkah tunggal. 
        # topologi terbalik(reversed) pendek akan memberikan urutan DAG.
        # Semula, kita akan mengulangnya melalui depth-first DAG, yang menyebabkan kesalahan
        # kalkulasi gradient saat input dihitung sebelum output-nya.
        def _topological_order():
            '''mengembalikan urutan topologi dependensi-nya.'''
            def _add_parents(node):
                if node not in visited:
                    visited.add(node)
                    for parent in node.parents:
                        _add_parents(parent)
                    ordered.append(node)

            ordered, visited = [], set()
            _add_parents(self)
            return ordered

        def _compute_grad_of_parents(node):
            '''beri node, hitung gradient parent: ∂output/∂parent = ∂output/∂node * ∂node/∂parent.'''
            for parent in node.parents:
                # pada saat _backward() di panggil node, kita sudah menghitung ∂output_∂node
                Δoutput_Δnode = node.grad  # PS: Python tidak support ∂ sebagai nama vaariabel :(
                
                # kita juga sudah menghitung ∂node_∂parent, saat node telah dibuat sebagai output operasi aritmatika
                Δnode_Δparent = node.grad_wrt[parent]
                
                # Kemudian, hitung dan simpan nilai ∂output/∂parent = ∂output/∂node * ∂node/∂parent
                #
                # sebenarnya += disini, sejak node dapat menjadi parent multiple doenstream node,
                # dan kita perlu mengakumulasi dengan benar semua gradient-nya. Lalu,
                # ∂output/∂parent = Σ_i ∂output/∂node_i * ∂node_i/∂parent
                #                    ^ untuk semua node_i yang mana itu adalah bagian dari parent downstream
                parent.grad += Δoutput_Δnode * Δnode_Δparent
                
        
        # ∂output/∂output = 1; untuk bootstraps backpropagation
        self.grad = 1
        
        # Telusuri grafik secara berurutan dari output ke input, dan hitung gradient!
        ordered = reversed(_topological_order())
        for node in ordered:
            _compute_grad_of_parents(node)
```

Kemudian, berikut hanya untuk menggambarkan grafik komputasi agar terlihat bagus:

```python
from graphviz import Digraph
     
def draw_graph(node):
    '''menggambarkan grafik dependensi node dengan grapviz. 
       Note: seperti  depth-first, lintasan sebelum diurutkan, tapi ini DAG lebih baik dari tree. 
             contohnya 1 node dapat digunakan pada multiple downstream node.'''
    def _draw_node(node):
        '''Menggambar / menambahkan single node kepada grafik.'''
        # jangan menambahkan duplikasi node kepada grafik.
        # contohnya jika kita mencapai node 2X dari 2 downstream node, hanya menambahkan 1
        if f'\t{id(node)}' in dot.body: return
        
        # menambahkan node dengan teks yang sesuai
        if node.parent_op is None:
            node_text = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
                <TR><TD>value = {node.value:.4f}</TD></TR>
                <TR><TD>grad = {node.grad:.4f}</TD></TR>
                <TR><TD BGCOLOR="#c9c9c9"><FONT FACE="Courier" POINT-SIZE="12">input</FONT></TD></TR>
            </TABLE>>'''
        else:
            node_text = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
                <TR><TD>value = {node.value:.4f}</TD></TR>
                <TR><TD>grad = {node.grad:.4f}</TD></TR>
                <TR><TD BGCOLOR="#c2ebff"><FONT COLOR="#004261" FACE="Courier" POINT-SIZE="12">{node.parent_op}</FONT></TD></TR>
            </TABLE>>'''
        dot.node(str(id(node)), node_text)
            
    def _draw_edge(parent, node):
        '''Menggambar / menambahkan sisi tunggal berarah kepada grafik (parent -> node).'''
        # Jangan menambahkan duplikasi tepi kepada grafik. 
        # contohnya jika kita mencapai node 2X dari 2 node, hanya menambahkan tepian kepada parent 1X
        if f'\t{id(parent)} -> {id(node)}' in dot.body: return
        
        # menambahkan edge/tepi
        dot.edge(str(id(parent)), str(id(node)))
    
    def _draw_parents(node):
        '''melintasi secara rekursif, menggambar parent pada langkah child (untuk menggambar tepi).'''
        for parent in node.parents:
            _draw_node(parent)
            _draw_edge(parent, node)
            _draw_parents(parent)
   
    dot = Digraph(graph_attr={'rankdir': 'BT'}, node_attr={'shape': 'plaintext'})
    _draw_node(node)     # menggambar root / output      
    _draw_parents(node)  # menggambar sisa grafik
    
    return dot
```


## Contoh Penggunaan

Dibawah ini adalah contoh demo secara fungsional (yang sedikit dibuat-buat) yang menunjukan sejumlah kemungkinan operasi yang didukung:

```python
from smallgrad import Scalar, draw_graph

# buat scalar
a = Scalar(1.5)

# buat sedikit perhitungan
b = Scalar(-4.0)
c = a**3 / 5
d = c + (b**2).relu()

# hitung gradient
d.backward()

# plot grafik komputasi
draw_graph(d)
```

output:

{:refdef: style="text-align: center;"}
<img src="/assets/smallgrad/smallgrad-1.svg">
{:refdef}

### Referensi:

[^1]: Andrej Karpathy ["micrograd"](https://github.com/karpathy/micrograd)