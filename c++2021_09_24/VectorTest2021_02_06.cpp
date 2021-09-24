/*
 * @Author: xudawu
 * @Date: 2021-01-23 15:25:29
 * vector 说明
   vector是向量类型，可以容纳许多类型的数据，因此也被称为容器
   (可以理解为动态数组，是封装好了的类）
   进行vector操作前应添加头文件#include <vector>
 */
#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;
class VectorTest2021_02_06
{
private:
public:
    void vectorTest();
    void showArray(int a[], int length);       //输出显示int数组
    void showArray(vector<int> a, int length); //输出显示vector数组
};

void VectorTest2021_02_06 ::vectorTest()
{
    vector<int> a;
    int b[8] = {2, 7, 8, 6, 1, 3, 4, 9};
    int length = sizeof(b) / sizeof(int); //sizeof()获取的是所占的地址大小，单位为字节(1B)，
    //函数传数组是传的是数组地址，不是数组空间，sizeof尽量不在函数内使用
    cout << "长度为" << length << "的b数组初值为:" << endl;
    showArray(b, length);
    for (int i = 0; i < length; i++) //用b数组给动态数组a赋值
    {
        a.push_back(b[i]);
    }
    cout << "用a.push_back(b[i]);给vector数组a赋值后a数组为" << endl;
    showArray(a, length);
    sort(a.begin(), a.end()); //a数组排序
    cout << "a数组用sort(a.begin(), a.end());排序后为:" << endl;
    showArray(a, length);
}
void VectorTest2021_02_06 ::showArray(int a[], int length)
{
    for (int i = 0; i < length; i++)
    {
        cout << a[i] << " ";
    }
    cout << endl;
}

void VectorTest2021_02_06 ::showArray(vector<int> a, int length)
{
    for (int i = 0; i < length; i++)
    {
        cout << a[i] << " ";
    }
    cout << endl;
}
/*一.vector初始化
vector<int> a(10);
//1.定义具有10个整型元素的向量（尖括号为元素类型名，它可以是任何合法的数据类型），不具有初值，其值不确定
vector<int> a(10,1);
//2.定义具有10个整型元素的向量，且给出的每个元素初值为1
vector<int> a(b);
//3.用向量b给向量a赋值，a的值完全等价于b的值
vector<int> a(b.begin(), b.begin + 3);
//4.将向量b中从0-2（共三个）的元素赋值给a，a的类型为int型
int b[7] = {1, 2, 3, 4, 5, 6, 7};
vector<int> a(b,b+7）;
//5.从数组中获得初值
*/

/*二.vector对象的常用内置函数使用（举例说明）
 #include<vector>
 vector<int> a,b;
（1）a.assign(b.begin(), b.begin()+3); //b为向量，将b的0~2个元素构成的向量赋给a
（2）a.assign(4,2); //是a只含4个元素，且每个元素为2
（3）a.back(); //返回a的最后一个元素
（4）a.front(); //返回a的第一个元素
（5）a[i]; //返回a的第i个元素，当且仅当a[i]存在
（6）a.clear(); //清空a中的元素
（7）a.empty(); //判断a是否为空，空则返回ture,不空则返回false
（8）a.pop_back(); //删除a向量的最后一个元素
（9）a.erase(a.begin()+1,a.begin()+3); //删除a中第1个（从第0个算起）到第2个元素，
    也就是说删除的元素从a.begin()+1算起（包括它）一直到a.begin()+3（不包括它）
（10）a.push_back(5); //在a的最后一个向量后插入一个元素，其值为5
（11）a.insert(a.begin()+1,5); //在a的第1个元素（从第0个算起）的位置插入数值5，
     如a为1,2,3,4，插入元素后为1,5,2,3,4
（12）a.insert(a.begin()+1,3,5); //在a的第1个元素（从第0个算起）的位置插入3个数，其值都为5
（13）a.insert(a.begin()+1,b+3,b+6); //b为数组，在a的第1个元素（从第0个算起）
     的位置插入b的第3个元素到第5个元素（不包括b+6），如b为1,2,3,4,5,9,8，插入元素后为1,4,5,9,2,3,4,5,9,8
（14）a.size(); //返回a中元素的个数；
（15）a.capacity(); //返回a在内存中总共可以容纳的元素个数
（16）a.resize(10); //将a的现有元素个数调至10个，多则删，少则补，其值随机
（17）a.resize(10,2); //将a的现有元素个数调至10个，多则删，少则补，其值为2
（18）a.reserve(100); //将a的容量（capacity）扩充至100，也就是说现在测试a.capacity();的时候返回值是100.
     这种操作只有在需要给a添加大量数据的时候才显得有意义，
     因为这将避免内存多次容量扩充操作（当a的容量不足时电脑会自动扩容，当然这必然降低性能） 
（19）a.swap(b); //b为向量，将a中的元素和b中的元素进行整体性交换
（20）a==b; //b为向量，向量的比较操作还有!=,>=,<=,>,<
*/

/* #include <algorithm>
（1）sort(a.begin(), a.end()); //对a中的从a.begin()（包括它）到a.end()（不包括它）的元素进行从小到大排列
（2）reverse(a.begin(), a.end()); //对a中的从a.begin()（包括它）到a.end()（不包括它）的元素倒置，但不排列，如a中元素为1,3,2,4,倒置后为4,2,3,1
（3）copy(a.begin(), a.end(), b.begin() + 1); //把a中的从a.begin()（包括它）到a.end()（不包括它）的元素复制到b中，从b.begin()+1的位置（包括它）开始复制，覆盖掉原有元素
（4）find(a.begin(), a.end(), 10); //在a中的从a.begin()（包括它）到a.end()（不包括它）的元素中查找10，若存在返回其在向量中的位置
 */