import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        
        self.data = np.array(data, dtype=np.float32)
        
        self.grad = np.zeros_like(self.data)
        
        self._backward = lambda: None
        
        self._prev = set(_children)

        self._op = _op

        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        # forward
        out = Tensor(self.data + other.data, _children=(self, other), _op='+')

        #backward
        def _backward():
            grad = out.grad

            if self.data.shape != grad.shape:
                self.grad += np.sum(grad, axis=0, keepdims=True)
            else:
                self.grad += out.grad

            if other.data.shape != grad.shape:
                other.grad += np.sum(grad, axis=0, keepdims=True)
            else:
                other.grad += out.grad
        
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        # forward
        out = Tensor(self.data * other.data, _children=(self, other), _op='*')

        #backward
        def _backward():
            grad = out.grad
            term_self = grad * other.data
            if self.data.shape != grad.shape:
                self.grad += np.sum(term_self, axis=0, keepdims=True)
            else:
                self.grad += out.grad * other.data
            
            term_other = grad * self.data
            if other.data.shape != grad.shape:
                other.grad += np.sum(term_other, axis=0, keepdims=True)
            else:
                other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        #forward
        out = Tensor(self.data @ other.data, _children=(self, other), _op='@')

        #backward
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (other * -1) 

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "MicroTorch only supports float/int power for now"

        out = Tensor(self.data ** other, _children=(self,), _op=f'**')
        def _backward():
            self.grad += out.grad * (other * self.data ** (other - 1))
        out._backward = _backward
        return out

    def sum(self):
        # 1. Forward
        out = Tensor(np.sum(self.data), _children=(self,), _op='sum')

        # 2. Backward
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        #z = 1/(1+np.exp(-Z))
        z = 1/(1+np.exp(-self.data))

        #forward
        out = Tensor(z, _children=(self,), _op='sigmoid')

        #backward
        def _backward():
               self.grad += out.grad * (z * (1-z))

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            print(node)
            node._backward() 
        

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.data.shape})"

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

if __name__ == "__main__":
    x = Tensor([1,2,3], label="x")
    y = Tensor([4,5,6], label="y")

    print(x)
    print(f"x data {x.data}")
    print(f"x grad {x.grad}")
    print(f"x _prev  {x._prev}")
