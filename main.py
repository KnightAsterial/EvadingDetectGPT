from customDetectGPT import get_score 


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
print(get_score([sent1]))

# Human text: Sexhow railway station was a railway station built to serve the hamlet of Sexhow in North Yorkshire, England. The station was on the North Yorkshire and Cleveland's railway line between and , which opened in 1857. The line was extended progressively until it met the Whitby & Pickering Railway at . Sexhow station was closed in 1954 to passengers and four years later to goods. The station was located south of Stockton, and west of Battersby railway station. History The station was opened in April 1857, when the line from Picton was opened up as far as . Mapping shows the station to have had three sidings in the goods yard, coal drops and a crane. The main station buildings were on the westbound (Picton direction) side of the station. The station was south of the village that it served, and was actually in the parish of Carlton in Cleveland, which has led to speculation that it was named Sexhow to avoid confusion with railway station, which was originally named Carlton.
# AI text: Sexhow railway station was a railway station located in the town of Sexhow, on the Cumbrian Coast Line in North West England. The station was opened by the Lancashire and Yorkshire Railway on 7 October 1870. It was closed to passengers on 5 January 1950, and to goods on 12 May 1965. The station building is now a private residence. There is a small amount of trackage remaining near the building, used currently by a local agricultural business.