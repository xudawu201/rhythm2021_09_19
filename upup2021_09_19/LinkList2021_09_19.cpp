/*
 * @Author: xudawu
 * @Date: 2021-09-19 22:53:41
 */
#include <iostream>
using std::cout;
using std::endl;
struct ListNode
{
    int intValue;        //结点值
    ListNode *next; //结点下一地址
};

class LinkList2021_09_19
{
private:
    /* data */
public:
    LinkList2021_09_19(/* args */);
    ~LinkList2021_09_19();
    ListNode *createLinkList(int intArray[], int intArraySize);
    void showLinkList(ListNode *linkList);
};

LinkList2021_09_19::LinkList2021_09_19(/* args */)
{
}

LinkList2021_09_19::~LinkList2021_09_19()
{
}

ListNode *LinkList2021_09_19::createLinkList(int intArray[], int intArraySize)
{
    ListNode *pHead = new ListNode;//建立头结点
    ListNode *pFront = pHead;//建立前结点，是移动结点
    for (int i = 0; i < intArraySize; i++)//建立链表并存入数据
    {
        ListNode *pRear = new ListNode;//后结点，存入数据
        pRear->intValue = intArray[i];//存值
        pRear->next = nullptr;//下一结点地址置空
        pFront->next = pRear;//链接结点
        pFront = pRear;//后移结点
    }
    return pHead->next;//舍弃头部空值结点返回头结点指针
}

void LinkList2021_09_19::showLinkList(ListNode *linkList)
{
    ListNode *pTemp = new ListNode;
    pTemp = linkList; //建立临时结点，不破坏头结点
    while (pTemp != nullptr)
    {
        cout << pTemp->intValue << " ";
        pTemp = pTemp->next;
    }
    cout << endl;
}