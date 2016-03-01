#include "./crunch.hpp"

using namespace crunch;

int_vector crunch::test1(int n) {
    //const int dims [] = {n,d};
    int_vector out(n);
    for(int i=0; i<n; i++){
        out[i] = n-i;
    }
    return out;
}

void crunch::test2(float_vector v) {
    for(int i = 0; i < v.size(); i++) {
        printf("v[%d] = %f\n", i, v(i));
    }
    if (v.ndim() == 3) {
        v.sub(1,1,2) = 43.69;
        printf("v[%d] = %f\n", 0, v.sub(1,1,2));
    } else {
        printf("2e testje gaat niet door, want dimensie is... ");
    }
    printf("%d\n", v.ndim());
}
