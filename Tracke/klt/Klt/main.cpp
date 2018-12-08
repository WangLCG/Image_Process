// main.cpp : Defines the entry point for the console application.
//

#include <stdio.h>  // printf
#include "KltExamples.h"

int main(int argc, char* argv[])
{
    // select which example to run here
    const int which = 2;

    // run the appropriate example
    switch (which) {
    case 1:  RunExample1();  break;
    case 2:  RunExample2();  break;
    case 3:  RunExample3();  break;
    case 4:  RunExample4();  break;  // Note:  example4 reads output from example 3
    case 5:  RunExample5();  break;
    default:  printf("There is no example number %d\n", which);
    }

    getchar();
    return 0;
}

