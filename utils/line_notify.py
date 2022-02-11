# https://qiita.com/akeome/items/e1e0fecf2e754436afc8
def send(notification_message):
    """
    LINEに通知する
    """
    import requests
    import os
    from dotenv import load_dotenv
    load_dotenv

    line_notify_token = os.getenv('LINE_TOKEN')
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)