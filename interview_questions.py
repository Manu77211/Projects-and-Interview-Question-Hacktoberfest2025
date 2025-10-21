"""
Common Coding Interview Problems - Python Solutions
This module contains solutions to frequently asked coding interview questions.
Each function includes detailed comments explaining the approach and complexity.
"""

def two_sum(nums, target):
    """
    LeetCode 1: Two Sum
    Given an array of integers nums and an integer target,
    return indices of the two numbers such that they add up to target.

    Args:
        nums (list): List of integers
        target (int): Target sum

    Returns:
        list: Indices of two numbers that add up to target

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []

def is_palindrome(s):
    """
    LeetCode 125: Valid Palindrome
    Given a string s, determine if it is a palindrome,
    considering only alphanumeric characters and ignoring cases.

    Args:
        s (str): Input string

    Returns:
        bool: True if palindrome, False otherwise

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    left, right = 0, len(s) - 1
    while left < right:
        # Skip non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1

        # Compare characters (case insensitive)
        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True

def longest_substring_without_repeating(s):
    """
    LeetCode 3: Longest Substring Without Repeating Characters
    Given a string s, find the length of the longest substring
    without repeating characters.

    Args:
        s (str): Input string

    Returns:
        int: Length of longest substring

    Time Complexity: O(n)
    Space Complexity: O(min(n, m)) where m is charset size
    """
    char_set = set()
    left = 0
    max_length = 0

    for right in range(len(s)):
        # Remove characters from left until no duplicate
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1

        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)

    return max_length

def merge_intervals(intervals):
    """
    LeetCode 56: Merge Intervals
    Given an array of intervals where intervals[i] = [starti, endi],
    merge all overlapping intervals.

    Args:
        intervals (list): List of intervals

    Returns:
        list: Merged intervals

    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if not intervals:
        return []

    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for current in intervals[1:]:
        last = merged[-1]
        # If current interval overlaps with last merged interval
        if current[0] <= last[1]:
            # Merge them
            last[1] = max(last[1], current[1])
        else:
            # Add current interval
            merged.append(current)

    return merged

def binary_search(nums, target):
    """
    Binary Search Implementation
    Given a sorted array of integers and a target value,
    return the index if the target is found. If not, return -1.

    Args:
        nums (list): Sorted list of integers
        target (int): Target value

    Returns:
        int: Index of target or -1

    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

def fibonacci_iterative(n):
    """
    Calculate nth Fibonacci number using iterative approach.

    Args:
        n (int): Position in Fibonacci sequence

    Returns:
        int: nth Fibonacci number

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b

    return b

def fibonacci_memoization(n, memo=None):
    """
    Calculate nth Fibonacci number using memoization (top-down DP).

    Args:
        n (int): Position in Fibonacci sequence
        memo (dict): Memoization dictionary

    Returns:
        int: nth Fibonacci number

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if memo is None:
        memo = {}

    if n in memo:
        return memo[n]

    if n <= 1:
        return n

    memo[n] = fibonacci_memoization(n - 1, memo) + fibonacci_memoization(n - 2, memo)
    return memo[n]

def quicksort(arr):
    """
    QuickSort implementation using Lomuto partition scheme.

    Args:
        arr (list): Array to sort

    Returns:
        list: Sorted array

    Time Complexity: O(n log n) average, O(n^2) worst case
    Space Complexity: O(log n) due to recursion
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[-1]
    left = []
    right = []

    for num in arr[:-1]:
        if num <= pivot:
            left.append(num)
        else:
            right.append(num)

    return quicksort(left) + [pivot] + quicksort(right)

def is_valid_parentheses(s):
    """
    LeetCode 20: Valid Parentheses
    Given a string containing just the characters '(', ')', '{', '}', '[' and ']',
    determine if the input string is valid.

    Args:
        s (str): Input string

    Returns:
        bool: True if valid, False otherwise

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in mapping:
            # Pop from stack if not empty, else use dummy value
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)

    return not stack

def max_subarray_sum(nums):
    """
    LeetCode 53: Maximum Subarray
    Given an integer array nums, find the contiguous subarray
    with the largest sum, and return its sum.

    Args:
        nums (list): Array of integers

    Returns:
        int: Maximum subarray sum

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not nums:
        return 0

    max_current = max_global = nums[0]

    for num in nums[1:]:
        max_current = max(num, max_current + num)
        if max_current > max_global:
            max_global = max_current

    return max_global

def main():
    """
    Main function demonstrating the interview problems.
    """
    print("=== Coding Interview Problems Demo ===\n")

    # Two Sum
    print("1. Two Sum Problem:")
    nums = [2, 7, 11, 15]
    target = 9
    result = two_sum(nums, target)
    print(f"   Input: {nums}, Target: {target}")
    print(f"   Result: {result}\n")

    # Valid Palindrome
    print("2. Valid Palindrome:")
    test_strings = ["A man, a plan, a canal: Panama", "race a car", "abba"]
    for s in test_strings:
        result = is_palindrome(s)
        print(f"   '{s}' -> {result}")
    print()

    # Longest Substring Without Repeating Characters
    print("3. Longest Substring Without Repeating Characters:")
    s = "abcabcbb"
    result = longest_substring_without_repeating(s)
    print(f"   Input: '{s}'")
    print(f"   Result: {result}\n")

    # Merge Intervals
    print("4. Merge Intervals:")
    intervals = [[1,3],[2,6],[8,10],[15,18]]
    result = merge_intervals(intervals)
    print(f"   Input: {intervals}")
    print(f"   Result: {result}\n")

    # Binary Search
    print("5. Binary Search:")
    nums = [-1,0,3,5,9,12]
    target = 9
    result = binary_search(nums, target)
    print(f"   Array: {nums}, Target: {target}")
    print(f"   Result: {result}\n")

    # Fibonacci
    print("6. Fibonacci Numbers:")
    n = 10
    iterative_result = fibonacci_iterative(n)
    memo_result = fibonacci_memoization(n)
    print(f"   F({n}) - Iterative: {iterative_result}")
    print(f"   F({n}) - Memoization: {memo_result}\n")

    # QuickSort
    print("7. QuickSort:")
    arr = [3, 6, 8, 10, 1, 2, 1]
    sorted_arr = quicksort(arr)
    print(f"   Input: {arr}")
    print(f"   Sorted: {sorted_arr}\n")

    # Valid Parentheses
    print("8. Valid Parentheses:")
    test_cases = ["()", "()[]{}", "(]", "([)]", "{[]}"]
    for case in test_cases:
        result = is_valid_parentheses(case)
        print(f"   '{case}' -> {result}")
    print()

    # Maximum Subarray Sum
    print("9. Maximum Subarray Sum:")
    nums = [-2,1,-3,4,-1,2,1,-5,4]
    result = max_subarray_sum(nums)
    print(f"   Array: {nums}")
    print(f"   Max Sum: {result}\n")

if __name__ == "__main__":
    main()
