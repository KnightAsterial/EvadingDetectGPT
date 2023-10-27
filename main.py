
def edit_distance(sent1, sent2):
    sent1_split = sent1.lower().split(' ')
    sent2_split = sent2.lower().split(' ')

    m, n = len(sent1_split), len(sent2_split)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Initialize the DP table
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if sent1_split[i - 1] == sent2_split[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i][j - 1] + 1, dp[i - 1][j] + 1)

    return dp[m][n]

sent1 = "Hello typing something to test out the code"
sent2 = "Hi tying to test out now the code"



sent1b = "AsdKj hi, test"
sent2b = 'asdkj hi test'

print(edit_distance(sent1, sent2))
print(edit_distance(sent1b, sent2b))