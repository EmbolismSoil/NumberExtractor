from sqlalchemy import  Column, String, create_engine, BigInteger
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from .config import config

Base = declarative_base()

class ClassCount(Base):
    __tablename__ = 'cls_cnt'
    cls = Column(String(64), primary_key=True)
    cnt = Column(BigInteger, index=True, nullable=False, default=0)


class ClassWordCount(Base):
    __tablename__ = 'cls_word_cnt'
    cls = Column(String(64), primary_key=True)
    word = Column(String(128), primary_key=True)
    cnt = Column(BigInteger, index=True, nullable=False, default=0)


engine = create_engine(config['db_url'])
DBSession = sessionmaker(bind=engine)