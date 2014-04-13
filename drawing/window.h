#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include "render.h"

class window : public QWidget
{
    Q_OBJECT
public:
    explicit window(QWidget *parent = 0);

signals:

public slots:

private:
    render *Render;

};

#endif // WINDOW_H
