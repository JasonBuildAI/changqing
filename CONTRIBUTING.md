# 参与 changqing

感谢你愿意花时间。这个仓库的规模不大，但有几条规矩是为了避免一类**很贵**的缺陷
——「看起来同步了，其实没有」。

## 开发环境

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## 提交之前必须全绿

```bash
ruff check .            # 静态检查
ruff format --check .   # 格式
mypy                    # 类型
python -m pytest        # 测试（默认档：不联网、不下模型、不启服务）
```

默认档**不许**引入网络依赖。需要真实 API Key 的用例请打 `@pytest.mark.online`，
它们不会被默认档选中。

## 提交信息

英文，Conventional Commits：

```
feat(store): add gzip archive for expired months
fix(retrieve): return empty instead of raising on timeout
refactor(extract): inject persona profile instead of hardcoding
docs: rewrite architecture guide
test: add host-coupling guard
chore: bump ruff to 0.7
```

一个提交只做一件事，能独立描述。**不要**把无关改动混在一起。

## 代码约定

- **注释讲「为什么」，不是「做了什么」。** 代码已经说了它做了什么。
- **同一句话只有一处权威定义。** 别处要么不写，要么写成指向它的指针。
  这个仓库踩过这个坑：一份分组清单曾在两处各抄一份，其中一处漏了一项，
  结果是「只有某一种记忆存在时，检索被静默挡掉」——不报错、不变红。
- **每加一条护栏都要问：它真的会失败吗？怎么证明？** 恒真断言、被 `xfail`
  吞掉的报错、猴补改不到的模块级拷贝，都出现过。新增护栏时请附一条
  「故意弄坏它、看它变红」的证据（可以写在 PR 描述里）。
- **核心包零第三方依赖。** 任何新的运行时依赖都必须挂到
  `[project.optional-dependencies]` 的某个 extras 上，并且缺失时要优雅降级。
- **不要往仓库里放真实对话数据。** 测试用固定夹具，写在 `tests/` 里。

## 新增一个「外部能力」适配器

嵌入模型与对话模型都通过 Protocol 注入（见 `src/changqing/ports.py`）。
新增适配器时：

1. 在 `src/changqing/adapters/` 下实现 Protocol；
2. 为它写一份**不联网**的单测（用 stub transport，不要真的发请求）；
3. 在 `docs/api.md` 的适配器矩阵里加一行。

## Pull Request

- 说明**为什么**要改，而不只是改了什么。
- 行为有变化就同时改文档——同一个 PR 里。
- 接口有变化请在描述里写清兼容性影响。

## 许可

提交即表示你同意以 Apache-2.0 授权你的贡献。

