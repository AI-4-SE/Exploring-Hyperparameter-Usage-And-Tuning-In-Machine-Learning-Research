#!/usr/bin/env python
# coding: utf-8

# ### Oi Let. Esse é o seu 5º desafio. Um pouco de diversão na mistura de Python com Futebol. Você pode olhar o código de outros Kagglers para ver como eles fizeram [aqui](https://www.kaggle.com/datasets/adaoduque/campeonato-brasileiro-de-futebol/code)
# 
# 1. Fazer um fork desse Notebook
# 2. Executar esse código Python
# 3. Analisar o relatário gerado pelo Pandas Profiling Report
# 4. Me explicar tudo que você entendeu sobre essa base.
# 5. Qual a variável com maior dispersão?
# 6. Quais as duas variáveis que possuem maior correlação de Cramér?
# 7. Os dados se referem à qual perído de tempo?
# 8. Eu acredito que a força dos times mandante diminuiu muito durante a pademia por causa dos jogos sem torcida. Será que os dados provam isso? Dica: Crie uma variável ano
# 9. Crie uma flag_pandemia para separar os dados da pamdemia dos demais períodos
# 10. Compare o percentual de vitória dos mandantes durante a pandemia com os demais anos
# 11. Compare o saldo de gols dos mandantes durante a pandemia com os demais anos
# 12. Usando essas duas métricas (% vitórias e saldo de gols) você tem evidências para provar que o desepenho dos mandantes foi inferior no perido de jogos sem o apoio da torcida?
# 13. Qual é a torcida que mais empura o seu time? Mostre qual foi o time que teve a maior queda de rendimento durante a pandemia.
# 14. Qual o time que mais melhorou seu rendimento jogando sem torcida?
# 15. Como ficou o rendimento do Bahia jogando sem torcida? Será que a torcida do Bahia faz mesmo a diferença ou é melhor jogar sem torcida? hahaha

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_profiling import ProfileReport
import pandasql as ps


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('/kaggle/input/campeonato-brasileiro-de-futebol/campeonato-brasileiro-full.csv')
df


# In[ ]:


#Generate report
#If the database has many records or columns, the report can take a long time
#If this is the case, disable the explorative, samples, correlations, missing_diagrams, duplicates and interactions options by commenting out
profile = ProfileReport(df, title=f"Pandas Profiling Report for df_prices"
                        ,explorative=True
                        ,samples=None
                        ,correlations={"cramers": {"calculate": False}}
                        ,missing_diagrams=None
                        ,duplicates=None
                        ,interactions=None
                       )
profile.to_file("profile.html")
display(profile)


# In[ ]:


#Você pode criar novas variáveis usando as variáveis existentes desse modo:
df["placar"]  =  df["mandante_placar"].map(str) + " x " + df["visitante_placar"].map(str)
df[["mandante_placar", "visitante_placar", "placar"]]


# In[ ]:


#você pode filtrar alguns registos da base dessa maneira:
df_2021 = df[(df['data'] >= '2021-01-01') & (df['data'] <= '2021-12-31')]
df_2021.head()


# In[ ]:


get_ipython().run_cell_magic('timeit', '', '#você pode agrupar os dados no pandas usando grupo by:\ndf[["mandante", "mandante_placar"]].groupby("mandante").sum("mandante_placar").sort_values(by=[\'mandante_placar\'], ascending=False)\n')


# In[ ]:


get_ipython().run_cell_magic('timeit', '', '#Se você não souber fazer alguma consulta usando pandas, você pode usar SQL\n#Com a biblioteca pandasql você consegue executar qualquer comando SQL em um dataframe pandas, como se ele fosse uma tabela em um banco de dados\n#A única desvantagem é que o pandasql é 100 vezes mais lento, então tome cuidade quando usar em bases muito grandes\n#Veja que o comando "%%timeit" calcula o tempo de execução do bloco. \n#O group by com pandas demorou 3.1 milissegundos, já o pandasql demorou 303 milissegundos.\nquery="""\nselect\n     mandante\n    ,sum(mandante_placar) as mandante_placar\nfrom df\ngroup by\n    mandante\norder by mandante_placar desc\n"""\nps.sqldf(query)\n')


# # Como ficou o desempenho dos mantandes durante a pandemia?

# In[ ]:




