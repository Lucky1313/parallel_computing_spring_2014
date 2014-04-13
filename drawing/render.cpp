#include "render.h"

#include <QPainter>

render::render(QWidget *parent) :
    QWidget(parent)
{
    setBackgroundRole(QPallate::Base);
    setAutoFillBackground(true);
}

QSize render::minimumSizeHint() const
{
    return QSize(100, 100);
}

QSize render::sizeHint() const
{
    return QSize(400, 200);
}
