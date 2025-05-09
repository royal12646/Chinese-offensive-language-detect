api_key = "sk-68e378376d854b05a4f330d9856320fc"
from http import HTTPStatus
from dashscope import Application
def call_agent_app(prompt):
    response = Application.call(app_id='0f1876c34e41430e9c088e16c7169d87',
                                prompt=prompt,
                                api_key=api_key,
                                workspace='llm-ow1vrv5uia77imdy',
                                )

    if response.status_code != HTTPStatus.OK:
        return
    else:
        result = response.output["text"]
    return result
