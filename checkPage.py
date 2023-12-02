# import docx
# def count_pages(docx_path):
#     # Load the .docx file
#     doc = docx.Document(docx_path)
#     # Define the default page size in inches (8.5 x 11 inches for letter size paper)
#     page_width = 8.5
#     page_height = 11
#     # Count the number of paragraphs that exceed the page size (indicating a new page)
#     page_count = 1  # Start with one page
#     current_height = 0
#     for paragraph in doc.paragraphs:
#         for run in paragraph.runs:
#             current_height += run.font.size
#             # Check if the current paragraph exceeds the page height
#             if current_height > page_height:
#                 print(page_count)
#                 current_height = 0
#     return page_count
# from typing import List
# # class Solution:
# #     def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
# #         def isAnagram(s: str, t: str) -> bool:
# #             if len(s) != len(t) or set(s) != set(t):
# #                 return False
# #             s_freq = {}
# #             t_freq = {}
# #             for char in s:
# #                 s_freq[char] = s_freq.get(char, 0) + 1
# #             for char in t:
# #                 t_freq[char] = t_freq.get(char, 0) + 1
# #             return s_freq == t_freq
# #         out = []
# #         copy_strs = strs.copy()
# #         # for i in range(0, len(copy_strs)):
# #         i = 0
# #         while len(copy_strs) != 0:
# #             out.append(
# #                 [copy_strs[0]]
# #             )
# #             copy_strs.pop(0)
# #             len_strs = len(copy_strs)
# #             j = 0
# #             for _ in range(0, len_strs):
# #                 if isAnagram(
# #                     copy_strs[j],
# #                     out[i][0]
# #                 ):
# #                     out[i].append(copy_strs[j])
# #                     copy_strs.pop(j)
# #                     len_strs -= 1
# #                 else:
# #                     j += 1
# #             i += 1
# #         return out
# from collections import defaultdict
# # class Solution:
# #     def productExceptSelf(self, nums: List[int]) -> List[int]:
# #         res = []
# #         for i in range(len(nums)):
# #             res.append(1)
# #             for j in range(len(nums)):
# #                 if i != j:
# #                     res[i] = res[i] * nums[j]
# #         return res
# from typing import Optional
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
#         def recursionInvertTree(root: Optional[TreeNode]):
#             if root is None:
#                 return
#             root.left, root.right = root.right, root.left
#             recursionInvertTree(root.left)
#             recursionInvertTree(root.right)
#         recursionInvertTree(root=root)
#         return root
# def out_tree(tree: Optional[TreeNode]):
#     if tree is None:
#         return
#     print(tree.val)
#     out_tree(tree.left)
#     out_tree(tree.right)
# a = Solution()
# new_root = a.invertTree(TreeNode([2, 1, 3]))
# out_tree(new_root)
# print(f"{'n':<10}\t{'h':<10}\t{'I':<10}")
# print(f"{n:<10}\t{h:<10.2e}\t{I_prev:<10.2e}")
# print(f"{n:<10}\t{h:<10.2e}\t{I_curr:<10.2e}")
