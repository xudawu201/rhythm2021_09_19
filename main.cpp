/*
 * @Author: xudawu
 * @Date: 2021-01-23 16:22:22
 */
#include <iostream>
#include "rhythm2021_02_06\VectorTest2021_02_06.cpp"
#include "rhythm2021_02_06\Student.cpp"
#include "rhythm2021_02_06\Pointer2021_09_07.cpp"
#include "rhythm2021_02_06\TwoSum2021_09_13.cpp"
#include "rhythm2021_02_06\LinkList2021_09_19.cpp"
#include "rhythm2021_02_06\AddTwoNumbers2021_09_18.cpp"
//using namespace std;//在std名称空间中使用
using std::cin; //只使需要使用的名称可用，cin()函数全称为std::cin(),使用using后可以省略std::写成cin()
using std::cout;
using std::endl;
int main()
{
    //using namespace std;//也可以采用在函数内使用名称空间的办法，不会影响到此函数外的函数
    int selection = 1;
    while (selection != 0)
    {
        cout << "<<<<<<<<<<<<<<<<<<<<<<<" << endl;
        cout << "输入数字选择功能序号" << endl;
        cout << "0.退出系统" << endl;
        cout << "1.vector测试" << endl;
        cout << "2.学生类测试" << endl;
        cout << "3.指针测试" << endl;
        cout << "4.leetcode1两数之和" << endl;
        cout << "5.单链表测试" << endl;
        cout << "6.leetcode2两数相加" << endl;
        cin >> selection;
        int b[8] = {2, 7, 8, 6, 1, 3, 4, 9};
        vector<int> a(b, b + 8);
        vector<int> c;
        switch (selection)
        {
        case 1:
        {
            VectorTest2021_02_06 v;
            v.vectorTest();
            break;
        }
        case 2:
        {
            Student student[2];
            for (int i = 0; i < 2; i++)
            {
                student[i].createStudent(i, "Aimer");
            }
            for (int i = 0; i < 2; i++)
            {
                student[i].showStudent();
            }

            break;
        }
        case 3:
        {
            Pointer2021_09_07 pointer;
            pointer.showPointertest();
            break;
        }
        case 4:
        {
            TwoSum2021_09_13 add;
            break;
        }
        case 5:
        {
            LinkList2021_09_19 linkList;
            ListNode *linkL = new ListNode;
            int intArray1[] = {3, 4, 7, 1, 2};
            int intArraySize1 = sizeof(intArray1) / sizeof(intArray1[0]);
            linkL = linkList.createLinkList(intArray1, intArraySize1);
            linkList.showLinkList(linkL);
            break;
        }
        case 6:
        {
            AddTwoNumbers2021_09_18 addTwoNumbers;
            addTwoNumbers.createTestCase();
            break;
        }
        default:
            break;
        }
    }
    cout << "hello world" << endl;
    return 0;
}