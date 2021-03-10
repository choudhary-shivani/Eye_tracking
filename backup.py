# fig = plt.figure()
# # sns.histplot(loaded_data, x='fixProp', shrink=.8, multiple="dodge",bins=10)
# ax = sns.boxplot(x=loaded_data['utag'], y=loaded_data['fixProp'])
# plt.savefig("Fix prop.png")
# fig = plt.figure()
# ax = sns.boxplot(x=loaded_data['utag'], y=loaded_data['crossreftime'])
# plt.savefig("cf_time.png")
# # fig = plt.figure()
# # ax = sns.scatterplot(data = loaded_data, x=loaded_data['fixProp'],
# #                      y=loaded_data[
# #     'crossreftime'],
# #                      hue='toklen')
# # ax = sns.pairplot(data=loaded_data[['crossreftime', 'fixProp', 'toklen']],
# #                   hue='toklen', kind='scatter', diag_kind='hist')
# plt.savefig("toklen.png")
# # fig = plt.figure()
# fig = plt.figure()
# ax = sns.scatterplot(x=loaded_data['utag'], y=loaded_data['GPT-FFD'])
# plt.show()
