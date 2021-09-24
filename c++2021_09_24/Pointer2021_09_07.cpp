/*
 * @Author: xudawu
 * @Date: 2021-09-07 17:16:25
 */
#include<iostream>
using namespace std;
class Pointer2021_09_07
{
private:
    /* data */
public:
    Pointer2021_09_07(/* args */);
    ~Pointer2021_09_07();
    void showPointertest();
};

Pointer2021_09_07::Pointer2021_09_07(/* args */)
{
}

Pointer2021_09_07::~Pointer2021_09_07()
{
}
void Pointer2021_09_07::showPointertest()
{
    int intNumber = 6;
    int *intPointerTest = &intNumber;// intPointerTest指向地址，*intPointerTest指向intPointerTest地址所存的内容
    cout << "*intPointerTest=:" << *intPointerTest << endl;
    cout << "intPointerTest=:" << intPointerTest << endl;
}