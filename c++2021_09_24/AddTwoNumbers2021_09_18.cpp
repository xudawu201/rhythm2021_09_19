/*
 * @Author: xudawu
 * @Date: 2021-09-18 14:41:42
 */

//Definition for singly-linked list.
#include<iostream>
using std::cout;
using std::endl;
struct ListNode1//在LinkList2021_09_19.cpp中定义了单链表结构，不能重复定义
{
    int val;        //结点值
    ListNode1 *next; //结点下一地址
    //ListNode1() : val(0), next(nullptr) {}
    //ListNode1(int x) : val(x), next(nullptr) {}
    //ListNode1(int x, ListNode1 *next) : val(x), next(next) {}
};

class AddTwoNumbers2021_09_18
{
private:
public:
    AddTwoNumbers2021_09_18(/* args */);
    ~AddTwoNumbers2021_09_18();
    ListNode1 *addTwoNumbers(ListNode1 *l1, ListNode1 *l2);
    ListNode1 *createLinkList(int intArray[],int intArraySize);
    void createTestCase();
};

AddTwoNumbers2021_09_18::AddTwoNumbers2021_09_18(/* args */)
{
}

AddTwoNumbers2021_09_18::~AddTwoNumbers2021_09_18()
{
}

ListNode1 *AddTwoNumbers2021_09_18::addTwoNumbers(ListNode1 *l1, ListNode1 *l2)
{
    ListNode1 *pResult = new ListNode1; //存放结果的链表，
    ListNode1 *pFront = pResult;       //移动指针
    int intCarry = 0;                 //进位
    int intSum = 0;                   //两数和
    while (l1 != nullptr || l2 != nullptr)
    {
        if (l1 != nullptr) //如果链表1不为空，将此结点值加上
        {
            intSum = intSum + l1->val;
            l1 = l1->next; //指针后移
        }
        if (l2 != nullptr) //如果链表2不为空，将此结点值加上
        {
            intSum = intSum + l2->val;
            l2 = l2->next; //指针后移
        }
        ListNode1 *pRear = new ListNode1;
        pRear->val = intSum % 10; //存值
        pRear->next = nullptr; //下一结点地址置空
        pFront->next = pRear;  //链接指针
        pFront = pRear;        //phead指针后移
        if (intSum >= 10)      //如果有进位,两数和初始化为1
        {
            intSum = 1;
        }
        else
        {
            intSum = 0; //无进位，两数和初始化为0
        }
    }
    if (intSum == 1) //如果两加数链表到达末尾且有进位
    {
        ListNode1 *pRear = new ListNode1;
        pRear->next = nullptr; //下一结点地址先置空
        pRear->val = 1;        //存进位1
        pFront->next = pRear;  //链接指针
    }
    return pResult->next;//指针后移舍弃头节点
}
ListNode1 *AddTwoNumbers2021_09_18::createLinkList(int intArray[], int intArraySize)
{
    ListNode1 *pHead = new ListNode1;
    ListNode1 *pFront = pHead;
    //int intArray[] = {3, 4, 7, 1, 2};
    for (int i = 0; i < intArraySize; i++)
    {
        ListNode1 *pRear = new ListNode1;
        pRear->val = intArray[i];
        pRear->next = nullptr;
        pFront->next = pRear;
        pFront = pRear;
    }
    return pHead->next;
}
void AddTwoNumbers2021_09_18::createTestCase()
{
    int intArray1[] = {3, 4, 7, 1, 2};
    int intArray2[] = {4, 6, 1, 2, 5};
    int intArraySize1 = sizeof(intArray1) / sizeof(intArray1[0]);
    int intArraySize2 = sizeof(intArray2) / sizeof(intArray2[0]);
    ListNode1 *phead1 = new ListNode1;//加数链表1
    ListNode1 *phead2 = new ListNode1;//加数链表2
    phead1 = createLinkList(intArray1, intArraySize1);
    phead2 = createLinkList(intArray2, intArraySize2);
    ListNode1 *pHeadResult = new ListNode1;//结果链表
    pHeadResult=addTwoNumbers(phead1, phead2);
    while (pHeadResult!=nullptr)
    {
        cout << pHeadResult->val << " ";
        pHeadResult = pHeadResult->next;
    }
    cout << endl;
}