import praw
from requests import Session

class RedditClient:
    """Reddit client
    """

    def __init__(self) -> None:
        self._session = Session()
        self._user_agent = "Mozilla/5.0 (Windows NT 6.1; rv:60.0) Gecko/20100101 Firefox/60.0"
        self._api = None

    def login(self, username, password, client_id, client_secret):
        self._api = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            username=username,
            password=password,
            user_agent = self._user_agent,
            requestor_kwargs={"session": self._session}
        )

    def logout(self):
        self._api = None

    @property
    def api(self):
        return self._api

    @property
    def is_logged_in(self):
        return self._api is not None

    def redditor(self, target_user):
        return self._api.redditor(target_user)

    def redditor_new_comments(self, target_user, limit=None):
        r = self.redditor(target_user)
        comments = list(r.comments.new(limit=limit))
        return [c.body.strip() for c in comments]

    def redditor_new_comments_raw(self, target_user, limit=None):
        r = self.redditor(target_user)
        return list(r.comments.new(limit=limit))
