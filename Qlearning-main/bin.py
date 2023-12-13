import connection as cn
from connection import get_state_reward

s = cn.connect(2037)

a,b = get_state_reward(s, "left")

print(a)
print(b)